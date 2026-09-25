// cold_fault.c - CPU cost of one decode step's cold PLE pages through the
// page cache, for the prefetch modes of the CPU worker and of TECHNIQUES.md #7.
//
// The test uses a private data file (not the served table), so it can evict
// its own pages with POSIX_FADV_DONTNEED and prove the eviction with mincore.
// The page offsets are uniform random. The table rows hash to uniform pages,
// so this is a proxy for the real row sets (an estimate, not a replay).
//
// Modes, per trial of N cold pages:
//   fault     touch the pages one at a time (one fault at a time). This is
//             the CPU floor of the GPU fault path with no prefetch.
//   fadvise   posix_fadvise(WILLNEED) per page, then touch all pages. This is
//             the shipped ple-io decode path (the Python loop is not included).
//   lead<us>  fadvise per page, wait <us> microseconds, then touch. This
//             models a prefetch that starts <us> before the gather.
//   pop<us>   fadvise per page, wait <us> microseconds, then one
//             MADV_POPULATE_READ per page (the page table entries of this
//             process are filled), then touch. The populate time is recorded.
//             A GPU that walks the host page tables (ATS) sees these pages
//             without a fault.
// For each trial the output records the issue time (the fadvise calls) and
// the stall time (the touch loop).
//
// Usage: cold_fault <file> <seed> <trials> <N> <lead_us,...>
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define PG 4096

static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}
static uint64_t rng;
static uint64_t next_u64(void) {
  uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static int cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
static double pct(double *v, int n, double p) { return v[(int)(p * (n - 1) + 0.5)]; }

static void report(const char *mode, int n, double *v, int trials, const char *what) {
  qsort(v, trials, sizeof(double), cmpd);
  double s = 0;
  for (int i = 0; i < trials; i++) s += v[i];
  printf("{\"mode\":\"%s\",\"n\":%d,\"what\":\"%s\",\"trials\":%d,\"mean\":%.1f,"
         "\"p50\":%.1f,\"p90\":%.1f,\"p99\":%.1f,\"max\":%.1f}\n",
         mode, n, what, trials, s / trials, pct(v, trials, .5), pct(v, trials, .9),
         pct(v, trials, .99), v[trials - 1]);
  fflush(stdout);
}

int main(int argc, char **argv) {
  if (argc < 6) return 2;
  int fd = open(argv[1], O_RDONLY);
  if (fd < 0) { perror("open"); return 1; }
  struct stat st;
  fstat(fd, &st);
  uint64_t npages = st.st_size / PG;
  rng = strtoull(argv[2], 0, 10);
  int trials = atoi(argv[3]), n = atoi(argv[4]);
  volatile uint8_t *map = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (map == MAP_FAILED) { perror("mmap"); return 1; }
  madvise((void *)map, st.st_size, MADV_RANDOM);  // no readahead around a fault
  uint64_t *pg = malloc(sizeof(uint64_t) * n);
  double *issue = malloc(sizeof(double) * trials), *stall = malloc(sizeof(double) * trials);
  double *popt = malloc(sizeof(double) * trials);
  unsigned char vec;
  long not_evicted = 0;

  // The mode list: "fault", "fadvise" (lead 0), then each lead value.
  char leads[256];
  snprintf(leads, sizeof leads, "fault,0,%s", argv[5]);
  char *save = NULL;
  for (char *tok = strtok_r(leads, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
    int is_fault = strcmp(tok, "fault") == 0;
    int is_pop = strncmp(tok, "pop", 3) == 0;
    double lead = is_fault ? 0 : atof(is_pop ? tok + 3 : tok);
    char mode[32];
    if (is_fault) snprintf(mode, sizeof mode, "fault");
    else if (is_pop) snprintf(mode, sizeof mode, "pop%.0f", lead);
    else if (lead == 0) snprintf(mode, sizeof mode, "fadvise");
    else snprintf(mode, sizeof mode, "lead%.0f", lead);
    for (int t = 0; t < trials; t++) {
      for (int i = 0; i < n; i++) {
        pg[i] = next_u64() % npages;
        // Drop the page table entry of this process first: the page cache
        // cannot evict a mapped page.
        madvise((void *)(map + pg[i] * PG), PG, MADV_DONTNEED);
        posix_fadvise(fd, pg[i] * PG, PG, POSIX_FADV_DONTNEED);
      }
      for (int i = 0; i < n; i++) {
        mincore((void *)(map + pg[i] * PG), PG, &vec);
        if (vec & 1) not_evicted++;
      }
      double t0 = now_us();
      if (!is_fault)
        for (int i = 0; i < n; i++) posix_fadvise(fd, pg[i] * PG, PG, POSIX_FADV_WILLNEED);
      double t1 = now_us();
      while (now_us() - t1 < lead) {}
      double tp = now_us();
      if (is_pop)
        for (int i = 0; i < n; i++)
          madvise((void *)(map + pg[i] * PG), PG, 22 /* MADV_POPULATE_READ */);
      double t2 = now_us();
      popt[t] = t2 - tp;
      uint64_t sum = 0;
      for (int i = 0; i < n; i++) sum += map[pg[i] * PG + 64];
      double t3 = now_us();
      if (sum == 0xFFFFFFFFFFFFULL) puts("");
      issue[t] = t1 - t0;
      stall[t] = t3 - t2;
      usleep(3000);
    }
    report(mode, n, issue, trials, "issue_us");
    report(mode, n, stall, trials, "stall_us");
    if (is_pop) report(mode, n, popt, trials, "populate_us");
  }
  printf("{\"not_evicted_pages\":%ld}\n", not_evicted);
  return 0;
}

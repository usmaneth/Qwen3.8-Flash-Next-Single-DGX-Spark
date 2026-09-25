// nvme_qd.c - measure the time to read N random 4 KiB pages of the PLE table
// with O_DIRECT at queue depth N (Linux native AIO), and at queue depth 1.
//
// Purpose: the prefetch floor for TECHNIQUES.md #7 and for R3b. O_DIRECT
// does not read from or write to the page cache, so this test does not change
// the residency of the served table.
//
// Usage: nvme_qd <file> <seed> <trials> <N> [<N> ...]
// Output: one JSON line per N, then one JSON line for queue depth 1.
// Build:  gcc -O2 -o nvme_qd nvme_qd.c
#define _GNU_SOURCE
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define PG 4096
#define MAXN 1024

static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static uint64_t rng;
static uint64_t next_u64(void) {  // splitmix64
  uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static int cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static double pct(double *v, int n, double p) {
  int i = (int)(p * (n - 1) + 0.5);
  return v[i];
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s file seed trials N...\n", argv[0]);
    return 2;
  }
  int fd = open(argv[1], O_RDONLY | O_DIRECT);
  if (fd < 0) { perror("open"); return 1; }
  struct stat st;
  fstat(fd, &st);
  uint64_t npages = st.st_size / PG;
  rng = strtoull(argv[2], 0, 10);
  int trials = atoi(argv[3]);

  aio_context_t ctx = 0;
  if (syscall(SYS_io_setup, MAXN, &ctx) < 0) { perror("io_setup"); return 1; }
  char *buf;
  if (posix_memalign((void **)&buf, PG, (size_t)MAXN * PG)) return 1;
  struct iocb cb[MAXN], *cbp[MAXN];
  struct io_event ev[MAXN];
  double *all = malloc(sizeof(double) * trials);
  double *first = malloc(sizeof(double) * trials);

  for (int a = 4; a < argc; a++) {
    int n = atoi(argv[a]);
    if (n < 1 || n > MAXN) continue;
    int errs = 0;
    for (int t = 0; t < trials; t++) {
      for (int i = 0; i < n; i++) {
        memset(&cb[i], 0, sizeof(cb[i]));
        cb[i].aio_fildes = fd;
        cb[i].aio_lio_opcode = IOCB_CMD_PREAD;
        cb[i].aio_buf = (uint64_t)(uintptr_t)(buf + (size_t)i * PG);
        cb[i].aio_nbytes = PG;
        cb[i].aio_offset = (int64_t)(next_u64() % npages) * PG;
        cbp[i] = &cb[i];
      }
      double t0 = now_us(), tf = -1;
      int sub = syscall(SYS_io_submit, ctx, n, cbp);
      if (sub != n) { perror("io_submit"); return 1; }
      int got = 0;
      while (got < n) {
        int r = syscall(SYS_io_getevents, ctx, 1, n - got, ev, NULL);
        if (r < 0) { perror("io_getevents"); return 1; }
        if (tf < 0) tf = now_us() - t0;
        for (int k = 0; k < r; k++) if ((long)ev[k].res != PG) errs++;
        got += r;
      }
      all[t] = now_us() - t0;
      first[t] = tf;
      usleep(2000);  // keep the load light: about 1 batch per 2 ms
    }
    qsort(all, trials, sizeof(double), cmpd);
    qsort(first, trials, sizeof(double), cmpd);
    double s = 0;
    for (int t = 0; t < trials; t++) s += all[t];
    printf("{\"mode\":\"qdN\",\"n\":%d,\"trials\":%d,\"errors\":%d,"
           "\"all_us\":{\"mean\":%.1f,\"p50\":%.1f,\"p90\":%.1f,\"p99\":%.1f,\"max\":%.1f},"
           "\"first_us_p50\":%.1f}\n",
           n, trials, errs, s / trials, pct(all, trials, .5), pct(all, trials, .9),
           pct(all, trials, .99), all[trials - 1], pct(first, trials, .5));
    fflush(stdout);
  }

  // Queue depth 1: one read at a time (the model of one fault at a time).
  int q1 = trials * 4;
  double *one = malloc(sizeof(double) * q1);
  for (int t = 0; t < q1; t++) {
    off_t off = (off_t)(next_u64() % npages) * PG;
    double t0 = now_us();
    if (pread(fd, buf, PG, off) != PG) { perror("pread"); return 1; }
    one[t] = now_us() - t0;
    if (t % 16 == 15) usleep(2000);
  }
  qsort(one, q1, sizeof(double), cmpd);
  double s = 0;
  for (int t = 0; t < q1; t++) s += one[t];
  printf("{\"mode\":\"qd1\",\"reads\":%d,\"us\":{\"mean\":%.1f,\"p50\":%.1f,\"p90\":%.1f,\"p99\":%.1f,\"max\":%.1f}}\n",
         q1, s / q1, pct(one, q1, .5), pct(one, q1, .9), pct(one, q1, .99), one[q1 - 1]);
  syscall(SYS_io_destroy, ctx);
  return 0;
}

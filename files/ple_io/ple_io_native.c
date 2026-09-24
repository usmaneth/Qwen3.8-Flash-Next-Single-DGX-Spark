// SPDX-License-Identifier: Apache-2.0
// Batched page prefetch for the PLE row gather (ple_io.py, backend "c").
//
// Build (lease step 0, with the glibc of the serving image):
//   docker run --rm --entrypoint cc -v $PWD:/s -w /s IMAGE
//     -O2 -shared -fPIC -pthread -o build/libple_io_native.so ple_io_native.c
//   (one command line)
//
// int64_t ple_prefetch(int fd, const int64_t *ids, int64_t n,
//                      int64_t row_width, int64_t total_pages, int threads)
//   Compute the first and the last 4 KiB page of each row, remove the
//   duplicate pages with a byte map (total_pages bytes, allocated one time),
//   and call posix_fadvise(fd, page << 12, 4096, POSIX_FADV_WILLNEED) for
//   each unique page. Above 4096 pages, `threads` pthreads share the pages.
//   Return the count of advised pages, or -errno. The call only gives advice:
//   the bytes that the caller reads next do not change.
//   The fd path takes no mmap_lock. The caller releases the GIL (ctypes).

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PAGE_SHIFT 12
#define POOL_MIN_PAGES 4096
#define MAX_THREADS 64

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static uint8_t *g_map = NULL;    // one byte per page of the table
static int64_t g_map_pages = 0;
static int64_t *g_list = NULL;   // unique pages of one call
static int64_t g_list_cap = 0;

struct job {
    int fd;
    const int64_t *pages;
    int64_t n;
    int err;
};

static void *run_job(void *arg) {
    struct job *j = (struct job *)arg;
    for (int64_t i = 0; i < j->n; i++) {
        int r = posix_fadvise(j->fd, (off_t)(j->pages[i] << PAGE_SHIFT),
                              1 << PAGE_SHIFT, POSIX_FADV_WILLNEED);
        if (r != 0 && j->err == 0) j->err = r;
    }
    return NULL;
}

static int ensure(int64_t total_pages, int64_t list_need) {
    if (g_map_pages < total_pages) {
        uint8_t *m = (uint8_t *)calloc((size_t)total_pages, 1);
        if (!m) return -ENOMEM;
        free(g_map);
        g_map = m;
        g_map_pages = total_pages;
    }
    if (g_list_cap < list_need) {
        int64_t cap = list_need < (1 << 18) ? (1 << 18) : list_need;
        int64_t *l = (int64_t *)malloc((size_t)cap * sizeof(int64_t));
        if (!l) return -ENOMEM;
        free(g_list);
        g_list = l;
        g_list_cap = cap;
    }
    return 0;
}

int64_t ple_prefetch(int fd, const int64_t *ids, int64_t n, int64_t row_width,
                     int64_t total_pages, int threads) {
    if (n <= 0) return 0;
    if (fd < 0 || !ids || row_width <= 0 || total_pages <= 0) return -EINVAL;
    pthread_mutex_lock(&g_lock);
    int64_t rc = ensure(total_pages, 2 * n);
    if (rc < 0) {
        pthread_mutex_unlock(&g_lock);
        return rc;
    }
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t off = ids[i] * row_width;
        int64_t p0 = off >> PAGE_SHIFT;
        int64_t p1 = (off + row_width - 1) >> PAGE_SHIFT;
        if (p0 < 0 || p1 >= total_pages) {
            rc = -EINVAL;
            break;
        }
        if (!g_map[p0]) { g_map[p0] = 1; g_list[k++] = p0; }
        if (p1 != p0 && !g_map[p1]) { g_map[p1] = 1; g_list[k++] = p1; }
    }
    // Clear only the set entries, so the next call starts from a zero map.
    for (int64_t i = 0; i < k; i++) g_map[g_list[i]] = 0;
    if (rc < 0) {
        pthread_mutex_unlock(&g_lock);
        return rc;
    }

    if (threads < 1) threads = 1;
    if (threads > MAX_THREADS) threads = MAX_THREADS;
    if (k <= POOL_MIN_PAGES) threads = 1;
    struct job jobs[MAX_THREADS];
    pthread_t tids[MAX_THREADS];
    int started[MAX_THREADS];
    int64_t step = (k + threads - 1) / threads;
    for (int t = 0; t < threads; t++) {
        int64_t lo = (int64_t)t * step;
        int64_t hi = lo + step > k ? k : lo + step;
        jobs[t].fd = fd;
        jobs[t].pages = g_list + lo;
        jobs[t].n = hi > lo ? hi - lo : 0;
        jobs[t].err = 0;
        started[t] = 0;
    }
    // The calling thread runs job 0. A thread that fails to start runs here.
    for (int t = 1; t < threads; t++) {
        if (jobs[t].n > 0 && pthread_create(&tids[t], NULL, run_job, &jobs[t]) == 0)
            started[t] = 1;
    }
    run_job(&jobs[0]);
    int err = jobs[0].err;
    for (int t = 1; t < threads; t++) {
        if (started[t]) pthread_join(tids[t], NULL);
        else if (jobs[t].n > 0) run_job(&jobs[t]);
        if (jobs[t].err && !err) err = jobs[t].err;
    }
    pthread_mutex_unlock(&g_lock);
    return err ? -(int64_t)err : k;
}

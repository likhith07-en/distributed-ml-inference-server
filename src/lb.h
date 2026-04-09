/* lb.h --- Load Balancer Interface (C)
 * Manages pool of Python inference workers.
 */
#ifndef LB_H
#define LB_H
#include <pthread.h>
#define MAX_WORKERS 16
typedef struct {
    char host[64];
    int  port;
    int  fd;        /* persistent TCP connection, -1 = disconnected */
    int  healthy;   /* 1 = healthy, 0 = down */
    int  active;    /* 1 = currently serving a request */
    long requests;  /* total requests served */
    long errors;    /* total errors */
} worker_t;
extern worker_t     workers[];
extern int          worker_count;
extern pthread_mutex_t lb_lock;
void lb_init(void);
int  lb_get_worker(void);
void lb_release_worker(int fd, int ok);
void lb_add_worker(const char *host, int port);
#endif /* LB_H */

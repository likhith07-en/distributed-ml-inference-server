/* lb.c --- Round-Robin Load Balancer (C) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netdb.h>
#include "lb.h"

worker_t workers[MAX_WORKERS];
int worker_count = 0;
static int rr_index = 0;
pthread_mutex_t lb_lock = PTHREAD_MUTEX_INITIALIZER;

static int connect_to_worker(const char *host, int port) {
    struct sockaddr_in addr;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct hostent *he = gethostbyname(host);
    if (!he) { close(fd); return -1; }
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
    struct timeval tv = { .tv_sec = 3, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd); return -1;
    }
    return fd;
}

void lb_init(void) {
    memset(workers, 0, sizeof(workers));
    lb_add_worker("127.0.0.1", 10001);
    lb_add_worker("127.0.0.1", 10002);
    lb_add_worker("127.0.0.1", 10003);
}

/* Returns fd via round-robin; -1 if no healthy worker */
int lb_get_worker(void) {
    pthread_mutex_lock(&lb_lock);
    for (int i = 0; i < worker_count; i++) {
        int idx = rr_index % worker_count;
        rr_index = (rr_index + 1) % worker_count;
        worker_t *w = &workers[idx];
        if (!w->healthy) continue;
        if (w->fd < 0) {
            w->fd = connect_to_worker(w->host, w->port);
            if (w->fd < 0) { w->healthy = 0; continue; }
        }
        w->active = 1; w->requests++;
        int fd = w->fd;
        pthread_mutex_unlock(&lb_lock);
        return fd;
    }
    pthread_mutex_unlock(&lb_lock);
    return -1; /* all workers down */
}

void lb_release_worker(int fd, int ok) {
    pthread_mutex_lock(&lb_lock);
    for (int i = 0; i < worker_count; i++) {
        if (workers[i].fd == fd) {
            workers[i].active = 0;
            if (!ok) {
                workers[i].errors++;
                close(workers[i].fd);
                workers[i].fd = -1; workers[i].healthy = 0;
            }
            break;
        }
    }
    pthread_mutex_unlock(&lb_lock);
}

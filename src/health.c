/* health.c --- Worker Health Monitor (C) */
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <arpa/inet.h>
#include "lb.h"
#define HEALTH_INTERVAL_SEC 5
#define CONNECT_TIMEOUT_SEC 2

static int probe_worker(const char *host, int port) {
    struct sockaddr_in addr;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;
    struct timeval tv = { .tv_sec = CONNECT_TIMEOUT_SEC };
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    inet_pton(AF_INET, host, &addr.sin_addr);
    int alive = connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0;
    close(fd);
    return alive;
}

static void *health_monitor_thread(void *arg) {
    (void)arg;
    while (1) {
        sleep(HEALTH_INTERVAL_SEC);
        pthread_mutex_lock(&lb_lock);
        for (int i = 0; i < worker_count; i++) {
            worker_t *w = &workers[i];
            int alive = probe_worker(w->host, w->port);
            if (!w->healthy && alive) {
                w->healthy = 1;
                if (w->fd >= 0) { close(w->fd); w->fd = -1; }
                printf("[Health] Worker %s:%d RECOVERED\n", w->host, w->port);
            } else if (w->healthy && !alive) {
                w->healthy = 0;
                if (w->fd >= 0) { close(w->fd); w->fd = -1; }
                printf("[Health] Worker %s:%d DOWN\n", w->host, w->port);
            }
        }
        pthread_mutex_unlock(&lb_lock);
    }
    return NULL;
}

void health_monitor_start(void) {
    pthread_t tid;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_create(&tid, &attr, health_monitor_thread, NULL);
    pthread_attr_destroy(&attr);
}

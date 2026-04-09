/* gateway.c --- C TCP API Gateway + Load Balancer
 * Build: gcc -Wall -O2 -pthread gateway.c lb.c health.c -o gateway
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "lb.h"
#define GATEWAY_PORT  9999
#define BACKLOG       256
#define MAX_PAYLOAD   (50*1024*1024)
#define CHUNK_SIZE    65536

static int recv_exact(int fd, uint8_t *buf, uint64_t n) {
    uint64_t r = 0;
    while (r < n) {
        ssize_t got = recv(fd, buf+r, n-r < CHUNK_SIZE ? n-r : CHUNK_SIZE, 0);
        if (got <= 0) return -1;
        r += got;
    }
    return 0;
}

static int send_exact(int fd, const uint8_t *buf, uint64_t n) {
    uint64_t s = 0;
    while (s < n) {
        ssize_t sent = send(fd, buf+s, n-s < CHUNK_SIZE ? n-s : CHUNK_SIZE, 0);
        if (sent <= 0) return -1;
        s += sent;
    }
    return 0;
}

typedef struct { int fd; struct sockaddr_in addr; } conn_ctx_t;

static void *handle_client(void *arg) {
    conn_ctx_t *ctx = (conn_ctx_t *)arg;
    int cfd = ctx->fd; free(ctx);
    uint8_t hdr[8]; uint64_t payload_len;
    uint8_t *img = NULL, *resp = NULL;
    int wfd = -1;
    if (recv_exact(cfd, hdr, 8) < 0) goto done;
    payload_len = be64toh(*(uint64_t *)hdr);
    if (!payload_len || payload_len > MAX_PAYLOAD) goto done;
    img = malloc(payload_len);
    if (!img || recv_exact(cfd, img, payload_len) < 0) goto done;
    wfd = lb_get_worker();
    if (wfd < 0) { /* send error JSON */ goto done; }
    { uint64_t nl = htobe64(payload_len);
      if (send_exact(wfd,(uint8_t*)&nl,8)<0 || send_exact(wfd,img,payload_len)<0)
          { lb_release_worker(wfd,0); wfd=-1; goto done; } }
    free(img); img = NULL;
    uint8_t rh[8];
    if (recv_exact(wfd,rh,8)<0) { lb_release_worker(wfd,0); wfd=-1; goto done; }
    uint64_t rl = be64toh(*(uint64_t*)rh);
    resp = malloc(rl+1);
    if (!resp || recv_exact(wfd,resp,rl)<0) { lb_release_worker(wfd,0); goto done; }
    lb_release_worker(wfd,1); wfd=-1;
    { uint64_t nr=htobe64(rl); send_exact(cfd,(uint8_t*)&nr,8); send_exact(cfd,resp,rl); }
done:
    free(img); free(resp);
    if (wfd>=0) lb_release_worker(wfd,0);
    close(cfd); return NULL;
}

int main(void) {
    lb_init(); health_monitor_start();
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt=1; setsockopt(srv,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
    struct sockaddr_in addr={.sin_family=AF_INET,.sin_addr.s_addr=INADDR_ANY,
                             .sin_port=htons(GATEWAY_PORT)};
    bind(srv,(struct sockaddr*)&addr,sizeof(addr));
    listen(srv,BACKLOG);
    printf("[GW] C Gateway listening on 0.0.0.0:%d\n",GATEWAY_PORT);
    while (1) {
        conn_ctx_t *ctx = malloc(sizeof(conn_ctx_t));
        socklen_t sl=sizeof(ctx->addr);
        ctx->fd = accept(srv,(struct sockaddr*)&ctx->addr,&sl);
        if (ctx->fd<0) { free(ctx); continue; }
        pthread_t tid; pthread_attr_t at;
        pthread_attr_init(&at);
        pthread_attr_setdetachstate(&at,PTHREAD_CREATE_DETACHED);
        pthread_create(&tid,&at,handle_client,ctx);
        pthread_attr_destroy(&at);
    }
}


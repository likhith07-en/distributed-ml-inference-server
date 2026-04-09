# Makefile — Build C Gateway components
CC     = gcc
CFLAGS = -Wall -Wextra -O2 -pthread -std=c11
SRCS   = src/gateway.c src/lb.c src/health.c
TARGET = gateway

all: $(TARGET)
$(TARGET): $(SRCS) src/lb.h
	$(CC) $(CFLAGS) -o $@ $(SRCS) -pthread

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

# Install Python production dependencies
install-py:
	pip install uvloop fastapi uvicorn pydantic prometheus-fastapi-instrumentator \
	    torch torchvision Pillow grpcio grpcio-tools prometheus-client \
	    aiohttp numpy scipy

# Build Docker images
docker-gateway:
	docker build -f Dockerfile.gateway -t inference-gateway:latest .
docker-worker:
	docker build -f Dockerfile.worker -t inference-worker:latest .

.PHONY: all clean run install-py docker-gateway docker-worker

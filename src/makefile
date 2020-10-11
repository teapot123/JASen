CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: margin joint

margin : margin.c
	$(CC) margin.c -o margin $(CFLAGS)

joint : joint.c
	$(CC) joint.c -o joint $(CFLAGS)

clean:
	rm -rf margin 
	rm -rf joint

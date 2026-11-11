CC     = gcc
CFLAGS = -std=c99 -Wall -Wextra -O2

all: wav_hist wav_cmp

wav_hist: wav_hist.c
	$(CC) $(CFLAGS) -o $@ $<

wav_cmp: wav_cmp.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f wav_hist wav_cmp

.PHONY: all clean

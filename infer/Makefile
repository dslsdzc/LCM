# LCM Inference Engine — Makefile
#
# Build targets:
#   make              — Release build (liblcm.a, liblcm.so)
#   make debug        — Debug build
#   make contracts    — Contract-checked build
#   make clean        — Remove build artifacts
#   make test         — Build and run unit tests
#
# Configurable:
#   make LCM_D=128    — Vector dimension (default: 256, matches lcm.h)
#
# Required: C99 compiler (gcc or clang), libm
#
# -lmvec: -march=native + -ffast-math make the compiler emit vectorized
# libmvec calls (e.g. _ZGVbN4v_logf). Without an explicit -lmvec the link is
# dropped by --as-needed (the symbol lives in libmvec, not libm) and the
# resulting .so fails to load via ctypes: undefined symbol.

CC       ?= gcc
CFLAGS   ?= -std=c99 -Wall -Wextra -Wpedantic -Werror
OPTFLAGS ?= -O2 -march=native -ffast-math
# Append (not ?=): a user-set LDFLAGS (e.g. -fuse-ld=mold) would otherwise
# silently replace these and the .so links without libm.
LDFLAGS  += -lm -lmvec

# Vector dimension (default matches lcm.h; override for old checkpoints)
LCM_D    ?= 256
CFLAGS   += -DLCM_D=$(LCM_D)

SRCS     = hyp.c lattice.c gvalue.c danger.c engine.c lcm_api.c
OBJS     = $(SRCS:.c=.o)

# ── Release build (default) ──────────────────────────────────────────────────
all: CFLAGS += $(OPTFLAGS) -DNDEBUG -fPIC
all: liblcm.a liblcm.so

# ── Debug build ──────────────────────────────────────────────────────────────
debug: CFLAGS += -g -O0 -DDEBUG -DLCM_USE_CONTRACTS -fPIC
debug: liblcm.a liblcm.so

# ── Contract-checked build ───────────────────────────────────────────────────
contracts: CFLAGS += $(OPTFLAGS) -DLCM_USE_CONTRACTS -fPIC
contracts: liblcm.a liblcm.so

liblcm.a: $(OBJS)
	$(AR) rcs $@ $(OBJS)

liblcm.so: $(OBJS)
	$(CC) -shared $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c lcm.h lcm_api.h
	$(CC) $(CFLAGS) -c -o $@ $<

# ── Test target ──────────────────────────────────────────────────────────────
test: CFLAGS += -g -O0 -DDEBUG -DLCM_USE_CONTRACTS
test: test_runner.c liblcm.a
	$(CC) $(CFLAGS) -o test_runner test_runner.c -L. -llcm $(LDFLAGS)
	LD_LIBRARY_PATH=. ./test_runner

# ── Clean ───────────────────────────────────────────────────────────────────
clean:
	rm -f *.o *.a *.so test_runner

.PHONY: all debug contracts clean test

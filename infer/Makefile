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
# No -ffast-math: it folds isnan/isfinite to constants, turns 0*inf and
# inf-inf into 0 (silently masking overflow), and disables the NaN guards in
# hyp.c/gvalue.c — measured with gcc 16.2.1. -O2 -march=native is enough.
#
# -lmvec: with -march=native the compiler may emit vectorized libmvec calls
# (e.g. _ZGVbN4v_logf). Without an explicit -lmvec the link is dropped by
# --as-needed (the symbol lives in libmvec, not libm) and the resulting .so
# fails to load via ctypes: undefined symbol.

CC       ?= gcc
CFLAGS   ?= -std=c99 -Wall -Wextra -Wpedantic -Werror
OPTFLAGS ?= -O2 -march=native
# Append (not ?=): a user-set LDFLAGS (e.g. -fuse-ld=mold) would otherwise
# silently replace these and the .so links without libm.
LDFLAGS  += -lm -lmvec

# Vector dimension (default matches lcm.h; override for old checkpoints)
LCM_D    ?= 256
CFLAGS   += -DLCM_D=$(LCM_D)

SRCS     = hyp.c lattice.c lattice_layers.c gvalue.c danger.c sha256.c engine.c loops_v3.c lcm_api.c
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

%.o: %.c lcm.h lcm_api.h lattice_internal.h
	$(CC) $(CFLAGS) -c -o $@ $<

# ── Test target ──────────────────────────────────────────────────────────────
test: CFLAGS += -g -O0 -DDEBUG -DLCM_USE_CONTRACTS -fPIC
test: test_runner.c liblcm.a liblcm.so
	$(CC) $(CFLAGS) -o test_runner test_runner.c -L. -llcm $(LDFLAGS)
	LD_LIBRARY_PATH=. ./test_runner

# ── JAX<->C parity build (task 11) ──────────────────────────────────────────
# The JAX parity harness runs at d=32 while the deployed engine is d=256, so
# this target builds a second .so in its own object directory. It never touches
# liblcm.so or the production *.o files.
#
#   make parity                      # liblcm_d32.so
#   make PARITY_D=64 parity          # liblcm_d64.so
PARITY_D      ?= 32
PARITY_OBJDIR  = parity_obj_d$(PARITY_D)
PARITY_OBJS    = $(SRCS:%.c=$(PARITY_OBJDIR)/%.o)
# Drop the global -DLCM_D=<deployed> so the parity build cannot silently pick
# up the wrong dimension when PARITY_D != LCM_D.
PARITY_CFLAGS  = $(filter-out -DLCM_D=%,$(CFLAGS)) \
                 -DLCM_D=$(PARITY_D) $(OPTFLAGS) -DNDEBUG -fPIC

liblcm_d$(PARITY_D).so: $(PARITY_OBJS)
	$(CC) -shared $(PARITY_CFLAGS) -o $@ $(PARITY_OBJS) $(LDFLAGS)

$(PARITY_OBJDIR)/%.o: %.c lcm.h lcm_api.h lattice_internal.h | $(PARITY_OBJDIR)
	$(CC) $(PARITY_CFLAGS) -c -o $@ $<

$(PARITY_OBJDIR):
	mkdir -p $@

parity: liblcm_d$(PARITY_D).so

# ── Clean ───────────────────────────────────────────────────────────────────
clean:
	rm -f *.o *.a *.so test_runner
	rm -rf parity_obj_d*
	rm -f liblcm_d*.so

.PHONY: all debug contracts clean test parity

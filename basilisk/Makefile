# Build the examples with qcc (Basilisk).
#   make BASILISK=/path/to/basilisk/src            planar_binary
# Third-party sandbox code is fetched with ./fetch_deps.sh (default into ./deps/basilisk-sandbox).
BASILISK ?= $(HOME)/basilisk/src
DEPS     ?= $(CURDIR)/deps/basilisk-sandbox
export BASILISK
export BASILISK_INCLUDE_PATH = $(DEPS)
QCC      = $(BASILISK)/qcc
QFLAGS   = -O2 -Wall -disable-dimensions
LIBS     = -L$(BASILISK)/gl -lfb_tiny -lglutils -lm

EXAMPLES = planar_binary ternary_planar crystal_frozenT thermosolutal_planar crystal3d

all: $(EXAMPLES)

%: examples/%.c mc_solver.h mc_physics.h
	cd examples && $(QCC) $(QFLAGS) $*.c -o $* $(LIBS)

check:                 # plain-C unit check of the physics layer (needs no Basilisk)
	gcc -O2 -Wall -DNC=2 tests/mc_physics_check.c -o /tmp/mc_physics_check -lm && /tmp/mc_physics_check

clean:
	rm -f $(addprefix examples/,$(EXAMPLES))
.PHONY: all check clean

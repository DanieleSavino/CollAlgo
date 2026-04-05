# ── Compiler ────────────────────────────────────────────────────────────────
CC := mpicc
AR := ar

# ── Directories ──────────────────────────────────────────────────────────────
SRC_DIR        := src
INC_DIR        := include
BENCH_SRC_DIR  := bench/src
BENCH_INC_DIR  := bench/include
BUILD_DIR      := build
OBJ_DIR        := $(BUILD_DIR)/obj
BENCH_OBJ_DIR  := $(BUILD_DIR)/obj/bench
LIB_DIR        := $(BUILD_DIR)/lib
BIN_DIR        := $(BUILD_DIR)/bin
VENDOR_DIR     := vendor/CollBench
VENDOR_LIB_DIR := $(VENDOR_DIR)/build/lib

# ── Library name ─────────────────────────────────────────────────────────────
LIB_NAME := collalgo

# ── Sources ───────────────────────────────────────────────────────────────────
LIB_SRCS   := $(wildcard $(SRC_DIR)/*.c)
BENCH_SRCS := $(wildcard $(BENCH_SRC_DIR)/*.c)

LIB_OBJS_RELEASE := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%_release.o, $(LIB_SRCS))
LIB_OBJS_PROFILE := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%_profile.o, $(LIB_SRCS))
LIB_OBJS_DEBUG   := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%_debug.o,   $(LIB_SRCS))
BENCH_OBJS       := $(patsubst $(BENCH_SRC_DIR)/%.c, $(BENCH_OBJ_DIR)/%.o, $(BENCH_SRCS))

# ── Common includes ───────────────────────────────────────────────────────────
COMMON_INC := -I$(INC_DIR) -I$(VENDOR_DIR)/include

# ── Library flags ─────────────────────────────────────────────────────────────
LIB_RELEASE_FLAGS := -Wall -Wextra -Wpedantic $(COMMON_INC) -O3 -march=native -DNDEBUG
LIB_PROFILE_FLAGS := -Wall -Wextra -Wpedantic $(COMMON_INC) -O3 -march=native -DNDEBUG -DCB_PROFILE
LIB_DEBUG_FLAGS   := -Wall -Wextra -Wpedantic $(COMMON_INC) -O0 -g3 -DDEBUG

# ── Bench flags ───────────────────────────────────────────────────────────────
BENCH_FLAGS   := -Wall -Wextra -Wpedantic $(COMMON_INC) -I$(BENCH_INC_DIR) -O3 -march=native -DNDEBUG
BENCH_LDFLAGS := -L$(LIB_DIR) -l$(LIB_NAME) -L$(VENDOR_LIB_DIR) -lcollbench -lm

PROFILE ?= 0
ifeq ($(PROFILE), 1)
    BENCH_FLAGS   += -DCB_PROFILE
    BENCH_LDFLAGS := -L$(LIB_DIR) -l$(LIB_NAME)_profile -L$(VENDOR_LIB_DIR) -lcollbench -lm
endif

# ── Targets ───────────────────────────────────────────────────────────────────
LIB_RELEASE := $(LIB_DIR)/lib$(LIB_NAME).a
LIB_PROFILE := $(LIB_DIR)/lib$(LIB_NAME)_profile.a
LIB_DEBUG   := $(LIB_DIR)/lib$(LIB_NAME)_debug.a
BENCH_BIN   := $(BIN_DIR)/bench

# ── Phony targets ─────────────────────────────────────────────────────────────
.PHONY: all lib lib-debug bench vendor clean clean-vendor compile_commands.json help

all: lib

lib: $(LIB_RELEASE)

lib-debug: $(LIB_DEBUG)

bench: $(BENCH_BIN)

clean-plot:
	rm -rf plots/

clean-all: clean clean-vendor

rebuild: clean lib

rebuild-all: clean-all
	$(MAKE) vendor
	$(MAKE) lib

rebench: clean bench

rebench-all: clean-all
	$(MAKE) vendor
	$(MAKE) bench

plot:
	./profile_np8.sbatch

replot: clean-plot
	$(MAKE) rebench PROFILE=1
	$(MAKE) plot

replot-all: clean-plot
	$(MAKE) rebench-all PROFILE=1
	$(MAKE) plot

ifeq ($(PROFILE), 1)
$(BENCH_BIN): $(LIB_PROFILE) $(BENCH_OBJS) | $(BIN_DIR)
else
$(BENCH_BIN): $(LIB_RELEASE) $(BENCH_OBJS) | $(BIN_DIR)
endif
	$(CC) $(BENCH_OBJS) -o $@ $(BENCH_LDFLAGS)

# ── Lib release objects ───────────────────────────────────────────────────────
$(OBJ_DIR)/%_release.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(LIB_RELEASE_FLAGS) -c $< -o $@

# ── Lib profile objects ───────────────────────────────────────────────────────
$(OBJ_DIR)/%_profile.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(LIB_PROFILE_FLAGS) -c $< -o $@

# ── Lib debug objects ─────────────────────────────────────────────────────────
$(OBJ_DIR)/%_debug.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(LIB_DEBUG_FLAGS) -c $< -o $@

# ── Archives ──────────────────────────────────────────────────────────────────
$(LIB_RELEASE): $(LIB_OBJS_RELEASE) | $(LIB_DIR)
	$(AR) rcs $@ $^

$(LIB_PROFILE): $(LIB_OBJS_PROFILE) | $(LIB_DIR)
	$(AR) rcs $@ $^

$(LIB_DEBUG): $(LIB_OBJS_DEBUG) | $(LIB_DIR)
	$(AR) rcs $@ $^

# ── Bench objects ─────────────────────────────────────────────────────────────
$(BENCH_OBJ_DIR)/%.o: $(BENCH_SRC_DIR)/%.c | $(BENCH_OBJ_DIR)
	$(CC) $(BENCH_FLAGS) -c $< -o $@

# ── Directory creation ────────────────────────────────────────────────────────
$(OBJ_DIR) $(BENCH_OBJ_DIR) $(LIB_DIR) $(BIN_DIR):
	mkdir -p $@

# ── Compile commands (for clangd / LSP) ──────────────────────────────────────
compile_commands.json: $(wildcard $(SRC_DIR)/*.c) $(wildcard $(BENCH_SRC_DIR)/*.c)
	$(MAKE) clean
	bear -- $(MAKE) lib-debug bench PROFILE=1

# ── Vendor ────────────────────────────────────────────────────────────────────
vendor:
	$(MAKE) -C $(VENDOR_DIR)

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD_DIR)

clean-vendor:
	$(MAKE) -C $(VENDOR_DIR) clean

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "Usage:"
	@echo "  make lib                     Build release lib"
	@echo "  make lib-debug               Build debug lib"
	@echo "  make bench [PROFILE=1]       Build bench executable"
	@echo "  make vendor                  Build vendor/CollBench"
	@echo "  make compile_commands.json   Regenerate for clangd (requires bear)"
	@echo "  make clean                   Remove build artifacts"
	@echo "  make clean-vendor            Remove vendor build artifacts"
	@echo ""
	@echo "Outputs:"
	@echo "  build/lib/libcollalgo.a          Release library"
	@echo "  build/lib/libcollalgo_profile.a  Profiled library (-DCB_PROFILE)"
	@echo "  build/lib/libcollalgo_debug.a    Debug library"
	@echo "  build/bin/bench                  Benchmark executable"

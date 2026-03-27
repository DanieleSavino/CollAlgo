# ── Compiler ────────────────────────────────────────────────────────────────
CC := mpicc

# ── Directories ──────────────────────────────────────────────────────────────
SRC_DIR        := src
INC_DIR        := include
BUILD_DIR      := build
OBJ_DIR        := $(BUILD_DIR)/obj
BIN_DIR        := $(BUILD_DIR)/bin
VENDOR_DIR     := vendor/CollBench
VENDOR_LIB_DIR := $(VENDOR_DIR)/build/lib

# ── Target ────────────────────────────────────────────────────────────────────
BIN_NAME := CollAlgo

# ── Sources ───────────────────────────────────────────────────────────────────
SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

# ── Flags ─────────────────────────────────────────────────────────────────────
COMMON_FLAGS := -Wall -Wextra -Wpedantic -I$(INC_DIR) -I$(VENDOR_DIR)/include

DEBUG_FLAGS   := $(COMMON_FLAGS) -O0 -g3 -DDEBUG
RELEASE_FLAGS := $(COMMON_FLAGS) -O3 -march=native -DNDEBUG

# ── Build mode (default: release) ─────────────────────────────────────────────
BUILD   ?= release
PROFILE ?= 0

ifeq ($(BUILD), debug)
    CFLAGS  := $(DEBUG_FLAGS)
    LDFLAGS := -L$(VENDOR_LIB_DIR) -lcollbench_debug -lm
    SUFFIX  := _debug
else ifeq ($(BUILD), release)
    CFLAGS  := $(RELEASE_FLAGS)
    LDFLAGS := -L$(VENDOR_LIB_DIR) -lcollbench -lm
    SUFFIX  :=
else
    $(error Unknown BUILD mode: $(BUILD). Use BUILD=debug or BUILD=release)
endif

ifeq ($(PROFILE), 1)
    CFLAGS += -DCB_PROFILE
endif

BIN_TARGET := $(BIN_DIR)/$(BIN_NAME)$(SUFFIX)

# ── Vendor ────────────────────────────────────────────────────────────────────

# ── Phony targets ─────────────────────────────────────────────────────────────
.PHONY: all debug release vendor clean clean-vendor help

all: $(BIN_TARGET)

debug:
	$(MAKE) BUILD=debug

release:
	$(MAKE) BUILD=release

vendor:
	$(MAKE) -C $(VENDOR_DIR)

# ── Compile objects ───────────────────────────────────────────────────────────
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# ── Link binary ───────────────────────────────────────────────────────────────
$(BIN_TARGET): $(OBJS) | $(BIN_DIR)
	$(CC) $^ -o $@ $(LDFLAGS)

# ── Directory creation ────────────────────────────────────────────────────────
$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

# ── Compile commands (for clangd / LSP) ──────────────────────────────────────
compile_commands.json: $(wildcard $(SRC_DIR)/*.c)
	$(MAKE) clean
	bear -- $(MAKE) BUILD=debug

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD_DIR) compile_commands.json

clean-vendor:
	$(MAKE) -C $(VENDOR_DIR) clean

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "Usage:"
	@echo "  make [BUILD=debug|release]   Build binary (default: release)"
	@echo "  make debug                   Shorthand for BUILD=debug"
	@echo "  make release                 Shorthand for BUILD=release"
	@echo "  make [PROFILE=1]             Enable CB_PROFILE"
	@echo "  make vendor                  Build vendor/CollBench"
	@echo "  make compile_commands.json   Regenerate for clangd (requires bear)"
	@echo "  make clean                   Remove build artifacts"
	@echo "  make clean-vendor            Remove vendor build artifacts"
	@echo ""
	@echo "Outputs:"
	@echo "  $(BIN_DIR)/$(BIN_NAME)[_debug]"

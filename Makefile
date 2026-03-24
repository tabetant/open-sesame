# ── Makefile — Voice Gate (Nios V, DE1-SoC) ──────────────────────────────────
#
# Usage:
#   make          → compile and link
#   make clean    → remove build artifacts
#   make load     → flash to board via niosv-download
#
# Run this on a DESL lab computer where the Nios V toolchain is installed.
# ─────────────────────────────────────────────────────────────────────────────

# ── Toolchain ────────────────────────────────────────────────────────────────
CC      = niosv-elf-gcc
OBJCOPY = niosv-elf-objcopy
SIZE    = niosv-elf-size

# ── Output ───────────────────────────────────────────────────────────────────
TARGET  = voice_gate
BUILD   = build

# ── Sources ──────────────────────────────────────────────────────────────────
SRCS =  main.c          \
        mfcc.c          \
        inference.c     \
        lego_motor.c    \
        output/model_data.c

OBJS = $(patsubst %.c, $(BUILD)/%.o, $(SRCS))

# ── Flags ────────────────────────────────────────────────────────────────────
# -O2 for everything except model_data.c (too many constants — slows GCC down)
CFLAGS_COMMON = -march=rv32im -mabi=ilp32 -Wall -I. -Ioutput
CFLAGS        = $(CFLAGS_COMMON) -O2
CFLAGS_DATA   = $(CFLAGS_COMMON) -O0   # no optimisation on weight table

LDFLAGS = -march=rv32im -mabi=ilp32 -T $(BSP_DIR)/linker.x \
          -nostdlib -lc -lm -lgcc

# ── BSP (Board Support Package) ───────────────────────────────────────────────
# The Monitor Program generates a BSP in the project directory.
# Point BSP_DIR at the folder that contains linker.x and crt0.o.
# Default: same directory as this Makefile.
BSP_DIR ?= .

# ── Rules ────────────────────────────────────────────────────────────────────
.PHONY: all clean load

all: $(BUILD)/$(TARGET).elf
	@$(SIZE) $<
	@echo ""
	@echo "Build complete: $(BUILD)/$(TARGET).elf"

# Link
$(BUILD)/$(TARGET).elf: $(OBJS)
	@mkdir -p $(BUILD)
	$(CC) $(OBJS) $(LDFLAGS) -o $@

# Compile — model_data.c gets -O0, everything else gets -O2
$(BUILD)/output/model_data.o: output/model_data.c output/model_data.h
	@mkdir -p $(BUILD)/output
	$(CC) $(CFLAGS_DATA) -c $< -o $@

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Flash to board (USB-Blaster must be connected)
load: $(BUILD)/$(TARGET).elf
	niosv-download -g $<

clean:
	rm -rf $(BUILD)

# CLAUDE.md — open-sesame

Project context and rules for Claude Code. Update this file as new information is gathered.

---

## What this project is

Voice-activated LEGO gate running on a **DE1-SoC board** with a **Nios V** soft processor (RISC-V, rv32im_zicsr).
ECE243 project at the University of Toronto.

**Pipeline:**
```
WM8731 audio codec (8 kHz, 16-bit I2S)
  → circular buffer (2 s / 16000 samples)
  → MFCC extraction (13 coefficients × 99 frames)
  → CNN inference
  → if P("open sesame") > 0.90 → open LEGO gate via motor
```

---

## Hardware

| Component | Detail |
|---|---|
| Board | DE1-SoC (Cyclone V FPGA) |
| Processor | Nios V (soft core, RISC-V) |
| Audio codec | WM8731 via I2C, 8 kHz, 16-bit |
| Motor | LEGO motor on GPIO (not yet wired as of this branch) |
| LEDs | 10× LEDR used as pipeline + confidence meter |
| VGA | Character buffer at `FPGA_CHAR_BASE` (80×60) |
| JTAG UART | `0xFF201000` — used for terminal debug output |

---

## File map

| File | Role | Touch? |
|---|---|---|
| `main.c` | Top-level pipeline, all hardware init, debug output | Yes |
| `mfcc.c / mfcc.h` | MFCC feature extraction | **No** |
| `inference.c / inference.h` | CNN forward pass | **No** |
| `lego_motor.c / lego_motor.h` | GPIO motor control + `delay()` | **No** |
| `output/model_data.c / .h` | Generated model weights — do not edit | **No** |
| `address_map.h` | All hardware base addresses | **No** |
| `input.c` | (unused / legacy) | No |
| `training/` | Python training scripts, not for board | No |

**All debug additions stay in `main.c` only**, unless a dedicated helper file is clearly needed.

---

## Toolchain

All tools are Windows `.exe` programs installed at:
```
C:/intelFPGA/QUARTUS_Lite_V23.1/
```

Key binaries (paths set in Makefile):
- Compiler: `riscv32-unknown-elf-gcc.exe`
- GDB server: `ash-riscv-gdb-server.exe`
- GDB client: `riscv32-unknown-elf-gdb.exe`
- JTAG terminal: `nios2-terminal.exe` (yes, nios2 — it works for Nios V too)
- Programmer: `quartus_pgm.exe`

The Makefile uses `cmd.exe` as SHELL and wraps everything in `cygwin64` bash for cross-compilation.
Running code is done entirely from **Windows PowerShell** on the DESL lab computers.

---

## Makefile targets

| Target | What it does |
|---|---|
| `./gmake COMPILE` | Compile all `.c` files → `main.elf` |
| `./gmake DE1-SoC` | Flash FPGA bitstream via JTAG (do once per power-on) |
| `./gmake GDB_SERVER` | Start the RISC-V GDB server (blocks the terminal) |
| `./gmake GDB_CLIENT` | Connect GDB, load ELF, set PC to `_start` |
| `./gmake TERMINAL` | Open `nios2-terminal.exe --instance 0` for JTAG UART output |
| `./gmake CLEAN` | Remove `.elf` and `.o` files |
| `./gmake DETECT_DEVICES` | List devices on the USB Blaster cable |
| `./gmake OBJDUMP` | Disassemble the ELF |
| `./gmake SYMBOLS` | Print symbol table |

---

## LED debug map (current)

| LED | Meaning |
|---|---|
| `LED[0]` | Boot complete, codec initialised |
| `LED[1]` | Pulses every ~0.1 s — audio samples flowing |
| `LED[2]` | Inference window ready (1 s of audio buffered) |
| `LED[3]` | MFCC extraction complete |
| `LED[4]` | CNN inference complete |
| `LED[5]` | `prob > 0.50` — weak match |
| `LED[6]` | `prob > 0.70` — moderate confidence |
| `LED[7]` | `prob > 0.80` — strong confidence |
| `LED[8]` | `prob > 0.90` — threshold met |
| `LED[9]` | `prob > 0.95` — very high confidence |

After each inference cycle LEDs reset to `LED[0]` only (idle).
If the program hangs, the last LED that stays lit indicates where it stopped.

---

## JTAG UART terminal output (current)

Messages printed to `nios2-terminal` during normal operation:
```
=== Open Sesame Gate ===
Audio codec initialised. LISTENING...

PROCESSING...
  MFCC extraction done
  CNN inference done
  Confidence: 0.87
  *** TRIGGERED! ***       ← only if prob > 0.90
LISTENING...
```

---

## VGA character buffer (current)

- Row 0: `=== Open Sesame Gate ===` (static title)
- Row 1: live status — cycles through `LISTENING...` / `PROCESSING...` / `MFCC done, running CNN...` / `Confidence: X.XX` / `*** TRIGGERED! ***`

---

## Gate trigger

The `open_and_close_gate()` function is implemented but the motor is **not yet wired**.
Detection is currently display-only (LEDs + JTAG print + VGA).
When the motor is wired, add the call back inside the `prob[1] > 0.90f` branch in `main.c`.

---

## JTAG index gotcha

`JTAG_INDEX_SoC` in the Makefile is currently `2`.
If `./gmake DE1-SoC` fails, run `./gmake DETECT_DEVICES` — if the two JTAG devices appear in reverse order, change `JTAG_INDEX_SoC` to `1`.

---

## Key constants

| Constant | Value | Meaning |
|---|---|---|
| `CIRC_BUF_SIZE` | 16000 | 2 s of audio at 8 kHz |
| `INFERENCE_STRIDE` | 8000 | Run inference every 1 s of new audio |
| `AUDIO_WINDOW_LEN` | 15880 | Samples fed to MFCC (`N_FFT + HOP*(N_FRAMES-1)`) |
| `N_MFCC` | 13 | MFCC coefficients |
| `N_FRAMES` | 99 | Time frames |
| `OPEN_DELAY` | 50000000 | Motor open duration (~0.5 s) |
| `HOLD_DELAY` | 20000000 | Gate hold-open duration |
| Confidence threshold | 0.90 | Minimum `prob[1]` to trigger |

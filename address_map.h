/*******************************************************************************
 * address_map.h — Memory-Mapped Peripheral Addresses for the DE1-SoC
 *
 * What is memory-mapped I/O?
 *   On the DE1-SoC, hardware peripherals (LEDs, audio, UART, …) are controlled
 *   by reading and writing to specific memory addresses — as if they were
 *   variables stored in RAM.  The FPGA's address decoder routes each address
 *   range to the right peripheral instead of to actual memory.
 *
 *   Example:
 *     volatile unsigned int *leds = (volatile unsigned int *)0xFF200000;
 *     *leds = 0x3FF;   // turns on all 10 red LEDs
 *
 *   The "volatile" keyword is required to tell the C compiler "do not cache
 *   this value in a register — the hardware may change it at any time".
 *
 * Address spaces:
 *   0x00000000 – 0x03FFFFFF : SDRAM (64 MB) — where your program and large arrays live
 *   0x08000000 – 0x0803FFFF : VGA pixel buffer (FPGA)
 *   0x09000000 – 0x09001FFF : VGA character buffer (80×60 text mode)
 *   0xFF200000 – 0xFF204FFF : Cyclone V FPGA peripherals (LEDs, switches, GPIO, …)
 *   0xFF700000 – 0xFFFFFFFF : HPS (ARM) peripherals (timers, I2C, …)
 *
 ******************************************************************************/

#ifndef __SYSTEM_INFO__
#define __SYSTEM_INFO__

#define BOARD "DE1-SoC"

/* ── Memory regions ──────────────────────────────────────────────────────── */

/* DDR3 SDRAM connected to the HPS (ARM) side — 1 GB, not used by Nios V */
#define DDR_BASE              0x40000000
#define DDR_END               0x7FFFFFFF

/* On-chip SRAM inside the Cyclone V — fast, small (~64 KB).
 * Used for the Nios V stack and time-critical code. */
#define A9_ONCHIP_BASE        0xFFFF0000
#define A9_ONCHIP_END         0xFFFFFFFF

/* SDRAM connected to the FPGA — 64 MB.
 * This is where all large arrays live: circ_buf, audio_window, model weights.
 * Your program's .text (code) and .data (globals) also go here. */
#define SDRAM_BASE            0x00000000
#define SDRAM_END             0x03FFFFFF

/* VGA pixel buffer — 640×480 pixels, 2 bytes per pixel (RGB565 colour).
 * Writing a colour value to address FPGA_PIXEL_BUF_BASE + 2*(y*640 + x)
 * draws that colour at pixel (x, y). */
#define FPGA_PIXEL_BUF_BASE   0x08000000
#define FPGA_PIXEL_BUF_END    0x0803FFFF

/* VGA character buffer — 80 columns × 60 rows of ASCII text.
 * Each cell is 2 bytes; only the low byte (ASCII code) is displayed.
 * Address of cell (row, col): FPGA_CHAR_BASE + 2*(row*80 + col)
 * Used in main.c to display status text without needing a framebuffer. */
#define FPGA_CHAR_BASE        0x09000000
#define FPGA_CHAR_END         0x09001FFF

/* ── Cyclone V FPGA peripherals ──────────────────────────────────────────── */

/* Red LEDs (LEDR[9:0]) — 10 LEDs controlled by bits [9:0] of a 32-bit register.
 * Write 0x001 to turn on only LED[0].  Write 0x3FF to turn on all 10. */
#define LED_BASE              0xFF200000
#define LEDR_BASE             0xFF200000  /* alias — same address */

/* 7-segment displays HEX3–HEX0 (four digits).
 * Each byte in the register drives one display (segment bitmask). */
#define HEX3_HEX0_BASE        0xFF200020

/* 7-segment displays HEX5–HEX4 (two more digits). */
#define HEX5_HEX4_BASE        0xFF200030

/* Slide switches SW[9:0] — read bits [9:0] to get current switch positions.
 * 1 = switch up, 0 = switch down. */
#define SW_BASE               0xFF200040

/* Pushbuttons KEY[3:0] — read bits [3:0] to get current button states.
 * Active-low: 0 = button pressed, 1 = button released.
 * KEY[0] is used in main.c to trigger the audio dump. */
#define KEY_BASE              0xFF200050

/* GPIO expansion headers JP1 and JP2 (40-pin headers).
 * Used in lego_motor.c to control the LEGO motor controller. */
#define JP1_BASE              0xFF200060
#define JP2_BASE              0xFF200070

/* PS/2 keyboard and mouse ports.
 * Not used in this project. */
#define PS2_BASE              0xFF200100
#define PS2_DUAL_BASE         0xFF200108

/* JTAG UART — serial port tunnelled through the USB Blaster cable.
 * Used in main.c to print debug text to nios2-terminal on the PC.
 * See the JTAG_UART_T struct in main.c for register details. */
#define JTAG_UART_BASE        0xFF201000

/* IrDA infrared transceiver — not used in this project. */
#define IrDA_BASE             0xFF201020

/* Interval timers — hardware countdown timers that can generate interrupts.
 * Not used in this project (we use delay() busy-loops instead). */
#define TIMER_BASE            0xFF202000
#define TIMER_2_BASE          0xFF202020

/* Audio codec I2C configuration controller.
 * Used in main.c to send configuration commands to the WM8731 codec.
 * See the I2C_T struct in main.c for register details. */
#define AV_CONFIG_BASE        0xFF203000

/* RGB resampler — converts video colour formats.
 * Not used in this project. */
#define RGB_RESAMPLER_BASE    0xFF203010

/* Pixel buffer controller — manages double-buffering for VGA pixel mode.
 * Not used in this project. */
#define PIXEL_BUF_CTRL_BASE   0xFF203020

/* Character buffer controller — manages the VGA text-mode buffer.
 * Not used in this project (we write directly to FPGA_CHAR_BASE instead). */
#define CHAR_BUF_CTRL_BASE    0xFF203030

/* WM8731 audio codec data registers (ADC input / DAC output FIFOs).
 * Used in main.c for all audio sampling and playback.
 * See the AUDIO_T struct in main.c for register details. */
#define AUDIO_BASE            0xFF203040

/* Video input capture — grabs frames from a camera.
 * Not used in this project. */
#define VIDEO_IN_BASE         0xFF203060

/* Edge detection controller — not used in this project. */
#define EDGE_DETECT_CTRL_BASE 0xFF203070

/* Analogue-to-digital converter — reads the 12-bit ADC inputs.
 * Not used in this project (we use the audio codec's ADC for sampling). */
#define ADC_BASE              0xFF204000

/* ── Cyclone V HPS (ARM) peripherals ─────────────────────────────────────── */
/* These are peripherals on the ARM hard processor side of the Cyclone V.
 * The Nios V soft processor can access them via the FPGA-to-HPS bridge,
 * but we do not use them in this project. */

#define HPS_GPIO1_BASE        0xFF709000
#define I2C0_BASE             0xFFC04000
#define I2C1_BASE             0xFFC05000
#define I2C2_BASE             0xFFC06000
#define I2C3_BASE             0xFFC07000
#define HPS_TIMER0_BASE       0xFFC08000
#define HPS_TIMER1_BASE       0xFFC09000
#define HPS_TIMER2_BASE       0xFFD00000
#define HPS_TIMER3_BASE       0xFFD01000
#define FPGA_BRIDGE           0xFFD0501C

#endif /* __SYSTEM_INFO__ */

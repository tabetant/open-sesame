/*
 * main.c — Voice-Activated LEGO Gate
 * ECE243 Project, University of Toronto
 *
 * ── What this file does ───────────────────────────────────────────────────────
 *
 * This is the top-level program that runs on the Nios V soft processor inside
 * the Cyclone V FPGA on the DE1-SoC board.  It does five things in a tight
 * loop:
 *
 *   1. Reads audio samples from the WM8731 codec (microphone input, 8 kHz).
 *   2. Writes them back to the DAC immediately so you can hear the mic through
 *      the speakers — this is called "passthrough" and is only here for
 *      debugging.
 *   3. Stores each sample in a circular (ring) buffer that always holds the
 *      most recent 5 seconds of audio.
 *   4. Every 3 seconds, extracts the last ~2 seconds of audio, runs MFCC
 *      feature extraction, then CNN inference, and lights LEDs to show the
 *      confidence that the phrase "open sesame" was just spoken.
 *   5. Lets you press KEY[0] to dump the raw audio buffer over the JTAG UART
 *      terminal so you can save it as a .wav file for retraining.
 *
 * ── Memory layout ─────────────────────────────────────────────────────────────
 *
 * All large arrays (circ_buf, audio_window, mfcc_buf, model weights) are
 * declared as static globals.  That puts them in SDRAM (64 MB), not on the
 * stack (which is only a few KB of on-chip SRAM).  If you ever see a hang
 * immediately after boot, a stack overflow from a large local array is the
 * first thing to check.
 *
 * ── Hardware pipeline ─────────────────────────────────────────────────────────
 *
 *  Microphone → WM8731 ADC → AUDIO FIFO → this polling loop
 *                                              ↓
 *                                        circ_buf (ring buffer)
 *                                              ↓   (every 3 s)
 *                                        audio_window (flat 2-s slice)
 *                                              ↓
 *                                        compute_mfcc()   → mfcc_buf [13][99]
 *                                              ↓
 *                                        run_inference()  → prob[2]
 *                                              ↓
 *                                        LEDs / JTAG UART / VGA
 */

#include "address_map.h"   /* all hardware base addresses (LED_BASE, AUDIO_BASE, …) */
#include "lego_motor.h"    /* setup_gpio(), spin_motor(), stop_motor(), delay()     */
#include "mfcc.h"          /* compute_mfcc() — 13 MFCCs × 99 frames                */
#include "inference.h"     /* run_inference() — CNN forward pass → prob[2]          */


/* ═══════════════════════════════════════════════════════════════════════════
 * AUDIO HARDWARE PERIPHERAL
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The DE1-SoC's audio IP core sits at AUDIO_BASE (0xFF203040).
 * Accessing it through a struct lets us refer to each register by name
 * instead of computing byte offsets by hand.
 *
 * The struct maps directly onto the hardware registers:
 *
 *   Offset 0x00 — control   : global enable / IRQ bits (we don't use IRQs)
 *   Offset 0x04 — rarc/ralc : how many samples are waiting to be read from
 *                              the right / left ADC FIFO
 *                              RARC = "Read samples Available, Right Channel"
 *                              RALC = "Read samples Available, Left  Channel"
 *   Offset 0x06 — wsrc/wslc : how many slots are free in the DAC output FIFO
 *                              WSRC = "Write Space available, Right Channel"
 *                              WSLC = "Write Space available, Left  Channel"
 *   Offset 0x08 — ldata     : reading pops the next LEFT  sample from ADC FIFO
 *                              writing pushes a sample into the LEFT  DAC FIFO
 *   Offset 0x0C — rdata     : same, but for the RIGHT channel
 *
 * What is a FIFO?
 *   FIFO = First In, First Out — a hardware queue.  The WM8731 codec samples
 *   the mic at 8 kHz and deposits each 16-bit value into the ADC FIFO.  The
 *   processor drains it at its own pace.  If the processor is slow (e.g.
 *   during MFCC computation), samples pile up and eventually get dropped once
 *   the FIFO fills.  The rarc/ralc bytes tell you how full it is.
 */
struct AUDIO_T {
    volatile unsigned int  control;  /* offset 0x00 — enable bits, not used here    */
    volatile unsigned char rarc;     /* offset 0x04 — ADC samples waiting (right)   */
    volatile unsigned char ralc;     /* offset 0x05 — ADC samples waiting (left)    */
    volatile unsigned char wsrc;     /* offset 0x06 — DAC free slots   (right)      */
    volatile unsigned char wslc;     /* offset 0x07 — DAC free slots   (left)       */
    volatile int           ldata;    /* offset 0x08 — read=ADC left, write=DAC left */
    volatile int           rdata;    /* offset 0x0C — read=ADC right,write=DAC right*/
};


/* ═══════════════════════════════════════════════════════════════════════════
 * I2C CONTROLLER
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * What is I2C?
 *   I2C (Inter-Integrated Circuit, pronounced "I-squared-C") is a simple
 *   two-wire serial bus for talking to chips like the WM8731 codec.
 *
 *   Wire 1 — SDA (Serial Data)  : the actual data bits go here
 *   Wire 2 — SCL (Serial Clock) : the master (us) pulses this to say "read now"
 *
 *   Every device on the bus has a 7-bit address (the WM8731 is 0x1A).
 *   A transaction looks like:
 *
 *     START → [device address] + WRITE bit → [byte 1] → [byte 2] → STOP
 *
 *   START and STOP are special electrical patterns on SDA/SCL that mark the
 *   beginning and end of a transaction.
 *
 * The I2C controller peripheral sits at AV_CONFIG_BASE (0xFF203000).
 * We configure it by writing to its registers:
 *
 *   prescale_lo / prescale_hi : clock divider — sets how fast SCL runs.
 *     Formula: prescale = (system_clock / (5 × desired_I2C_freq)) − 1
 *     At 50 MHz system clock, for 400 kHz I2C:
 *       prescale = (50,000,000 / (5 × 400,000)) − 1 = 25 − 1 = 24
 *     This value is split across two 8-bit registers: lo = 24, hi = 0.
 *
 *   control : bit 7 = enable the I2C core (write 0x80 to turn it on)
 *
 *   data    : byte to transmit (load this first, then trigger via cmd_status)
 *
 *   cmd_status : writing here starts a transfer; reading here shows status.
 *     Write bits:
 *       bit 7 (STA) — generate a START condition before this byte
 *       bit 6 (STO) — generate a STOP  condition after  this byte
 *       bit 4 (WR)  — write the byte in the data register onto the bus
 *     Read bits:
 *       bit 1 (TIP) — Transfer In Progress — spin on this to know when done
 *       bit 7 (RXACK) — 0 = device acknowledged, 1 = no-ack (error)
 */
struct I2C_T {
    volatile unsigned int prescale_lo;  /* 0x00 — low  byte of clock prescaler    */
    volatile unsigned int prescale_hi;  /* 0x04 — high byte of clock prescaler    */
    volatile unsigned int control;      /* 0x08 — bit 7 = core enable             */
    volatile unsigned int data;         /* 0x0C — byte to send / byte received    */
    volatile unsigned int cmd_status;   /* 0x10 — write=command, read=status      */
};


/* ═══════════════════════════════════════════════════════════════════════════
 * JTAG UART
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The JTAG UART is how we print text to the terminal on the PC.
 * "JTAG" refers to the same USB cable used to program the FPGA — the UART
 * (serial port) is tunnelled through that same connection.
 * Run `nios2-terminal.exe --instance 0` on the PC to see the output.
 *
 * The peripheral has just two 32-bit registers at JTAG_UART_BASE (0xFF201000):
 *
 *   data    [offset 0x00]:
 *     WRITE: bits [7:0]  = the byte you want to transmit
 *     READ : bits [7:0]  = the received byte (we don't use RX here)
 *            bit  [15]   = RVALID — 1 if there is a byte to read
 *            bits [31:16]= RAVAIL — how many more RX bytes are queued
 *
 *   control [offset 0x04]:
 *     bits [31:16] = WSPACE — how many free slots are in the TX FIFO
 *     bit  [1]     = WI     — interrupt enable for TX (we poll instead)
 *     bit  [0]     = RI     — interrupt enable for RX
 *
 * What is a TX FIFO?
 *   TX = Transmit.  FIFO = First In First Out (a hardware queue).
 *   When you write a byte to the data register, it goes into the FIFO.
 *   The hardware drains the FIFO over the USB link at whatever rate the
 *   PC can receive.  WSPACE tells you how many more bytes you can queue
 *   before the FIFO is full.  If WSPACE == 0, you must wait before writing
 *   another byte or it will be dropped.  jtag_putc() spins until there is
 *   room, then writes — this guarantees every byte gets through.
 *
 *   The FIFO size is 64 bytes.  At typical USB Blaster speeds you can
 *   sustain ~50–100 KB/s.  Dumping 40000 samples (5 s of audio) as ASCII
 *   integers (~7 chars each) = ~280 KB, which takes roughly 3–6 seconds.
 */
struct JTAG_UART_T {
    volatile unsigned int data;    /* [7:0]=byte to TX; [15]=RVALID; [31:16]=RAVAIL */
    volatile unsigned int control; /* [31:16]=WSPACE (free TX slots); [1]=WI; [0]=RI */
};

/* Pointer to the JTAG UART hardware registers */
static struct JTAG_UART_T *jtag = (struct JTAG_UART_T *)JTAG_UART_BASE;

/*
 * jtag_putc — send one ASCII character to the terminal.
 *
 * We first spin-wait until the TX FIFO has at least one free slot (WSPACE > 0),
 * then write the character into the data register.  The hardware takes care
 * of the rest.  This never drops a character, but it will stall the processor
 * if the FIFO is full.  That is fine here because we only print short strings
 * during inference (not inside the audio sample loop).
 */
static void jtag_putc(char c)
{
    /* Wait until there is room in the transmit FIFO.
     * control >> 16 extracts the upper 16 bits (WSPACE).
     * Loop until WSPACE is non-zero (at least one slot free). */
    while (((jtag->control >> 16) & 0xFFFF) == 0);

    /* Write the character.  Cast to unsigned char to strip the sign bit
     * so that e.g. '\n' (0x0A) is not sign-extended to 0xFFFFFF0A. */
    jtag->data = (unsigned char)c;
}

/*
 * jtag_puts — send a null-terminated C string to the terminal.
 * Loops over every character in the string and calls jtag_putc for each.
 * Use "\r\n" for line endings (the terminal needs both carriage-return
 * AND newline to move to the start of the next line).
 */
static void jtag_puts(const char *s)
{
    while (*s) jtag_putc(*s++);
}

/*
 * jtag_put_int — print a signed integer as decimal ASCII digits.
 *
 * We cannot use printf() or sprintf() because we are not linking against the
 * standard C library (it would make the ELF too large for SDRAM / JTAG load
 * time).  So we roll our own.
 *
 * Algorithm:
 *   1. Handle zero as a special case (the loop below would emit nothing).
 *   2. If negative, emit '-' and negate.
 *   3. Extract digits right-to-left by repeatedly taking n % 10, storing
 *      each digit in a temporary buffer.
 *   4. Print the buffer in reverse so the digits come out left-to-right.
 *
 * The buffer needs at most 6 chars for a 16-bit sample (-32768 → 6 digits).
 * We declare 7 to be safe with the null terminator if anyone adds one later.
 *
 * This is used by dump_audio_window() to serialise raw audio samples as
 * human-readable integers over the JTAG UART.
 */
static void jtag_put_int(int n)
{
    char buf[7];   /* enough for -32768 (6 chars) — no null needed, just printed */
    int  i = 0;

    if (n == 0) { jtag_putc('0'); return; }  /* special case: zero emits "0" */
    if (n < 0)  { jtag_putc('-'); n = -n; }  /* emit minus sign, work with positive */

    /* Build digits in REVERSE order (least-significant first) */
    while (n > 0) {
        buf[i++] = '0' + (char)(n % 10);  /* digit as ASCII: '0'=48, '9'=57 */
        n /= 10;
    }

    /* Print digits in FORWARD order by walking buf backwards */
    while (i > 0) jtag_putc(buf[--i]);
}

/*
 * jtag_put_float — print a float as "X.XX" (e.g. "0.87" or "1.00").
 *
 * This only works correctly for values in [0.00, 1.00], which is exactly the
 * range of CNN softmax probabilities.  No libc needed.
 *
 * How it works:
 *   whole = integer part (0 or 1 for probabilities)
 *   frac  = first two decimal places, computed as:
 *             (f - whole) × 100, rounded to nearest integer
 *   Guard: if rounding pushes frac to 100 (e.g. 0.999 → frac=100),
 *   roll over to whole+1 and frac=0, printing "1.00".
 */
static void jtag_put_float(float f)
{
    int whole = (int)f;
    int frac  = (int)((f - (float)whole) * 100.0f + 0.5f); /* round to 2 dp */
    if (frac >= 100) { whole++; frac = 0; }  /* handle 0.999… rounding up */
    jtag_putc('0' + (char)whole);            /* integer digit */
    jtag_putc('.');
    jtag_putc('0' + (char)(frac / 10));      /* tenths digit  */
    jtag_putc('0' + (char)(frac % 10));      /* hundredths digit */
}


/* ═══════════════════════════════════════════════════════════════════════════
 * VGA CHARACTER BUFFER
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The DE1-SoC's VGA controller includes a character buffer at FPGA_CHAR_BASE
 * (0x09000000).  Think of it like a terminal with 80 columns × 60 rows.
 * Each position on screen corresponds to a memory location, and writing an
 * ASCII code there makes that character appear.
 *
 * Memory layout:
 *   Each cell occupies 2 bytes (a 16-bit word).
 *   Only the LOW byte (bits [7:0]) is the ASCII character code.
 *   The HIGH byte can hold colour attributes but we leave it at zero
 *   (which gives white-on-black in the default colour scheme).
 *
 *   Address of cell (row, col):
 *     FPGA_CHAR_BASE + 2 × (row × 80 + col)
 *
 *   Example: row=1, col=5 → offset = 2×(1×80+5) = 2×85 = 170 bytes from base.
 *
 * We use:
 *   Row 0 — static title "=== Open Sesame Gate ===" (set at boot, never changed)
 *   Row 1 — live status line (LISTENING / PROCESSING / Confidence: X.XX / TRIGGERED)
 */
#define VGA_COLS  80   /* number of character columns on screen */
#define VGA_ROWS  60   /* number of character rows on screen    */

/* Cast the base address to a char pointer so we can write ASCII bytes to it */
static volatile char *vga_char_buf = (volatile char *)FPGA_CHAR_BASE;

/*
 * vga_clear — blank the entire screen by writing space characters to every cell.
 * Called once at boot before printing the title row.
 */
static void vga_clear(void)
{
    int r, c;
    for (r = 0; r < VGA_ROWS; r++)
        for (c = 0; c < VGA_COLS; c++)
            *(vga_char_buf + 2 * (r * VGA_COLS + c)) = ' ';  /* ASCII 32 = space */
}

/*
 * vga_puts_at — write a string starting at (row, col).
 * Stops when the string ends OR when we reach the right edge of the screen
 * (col == VGA_COLS), whichever comes first — no wraparound.
 */
static void vga_puts_at(int row, int col, const char *s)
{
    while (*s && col < VGA_COLS) {
        /* +2*(row*80+col) gives the byte address of the ASCII byte of this cell */
        *(vga_char_buf + 2 * (row * VGA_COLS + col)) = *s++;
        col++;
    }
}

/*
 * vga_status — update one row of the status display.
 *
 * First clears the entire row to spaces (prevents ghost characters from a
 * longer previous string lingering on screen), then writes the new string
 * from column 0.  This is the main function used to update row 1.
 */
static void vga_status(int row, const char *s)
{
    int c;
    /* Erase the row */
    for (c = 0; c < VGA_COLS; c++)
        *(vga_char_buf + 2 * (row * VGA_COLS + c)) = ' ';
    /* Write new content */
    vga_puts_at(row, 0, s);
}

/*
 * vga_put_float_at — write "X.XX" at a specific (row, col) position.
 *
 * Used to append the confidence value after "Confidence: " on row 1.
 * Same rounding logic as jtag_put_float.  Writes 4 characters total.
 */
static void vga_put_float_at(int row, int col, float f)
{
    int whole = (int)f;
    int frac  = (int)((f - (float)whole) * 100.0f + 0.5f);
    if (frac >= 100) { whole++; frac = 0; }
    char buf[5];
    buf[0] = '0' + (char)whole;
    buf[1] = '.';
    buf[2] = '0' + (char)(frac / 10);
    buf[3] = '0' + (char)(frac % 10);
    buf[4] = '\0';   /* null terminator so vga_puts_at knows where to stop */
    vga_puts_at(row, col, buf);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * I2C HELPER FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════════════ */

/* I2C command bits — combined into the cmd_status register to trigger actions */
#define I2C_START    0x90   /* STA (bit 7) + WR (bit 4) — start + write this byte */
#define I2C_WRITE    0x10   /* WR  (bit 4) only — write this byte (no start)      */
#define I2C_STOP     0x40   /* STO (bit 6) — generate stop condition after byte   */
#define I2C_RXACK    0x80   /* status bit — 0=device acked, 1=no-ack (error)      */
#define I2C_TIP      0x02   /* status bit — 1=transfer in progress, 0=done        */
#define I2C_BUSY     0x40   /* status bit — bus is busy (another master using it) */

/* WM8731 codec's I2C address.  CSB pin is tied low on the DE1-SoC board,
 * so the 7-bit address is 0x1A.  When shifted left by 1 and OR'd with 0
 * it becomes 0x34 (write mode). */
#define WM8731_ADDR  0x1A

/* Pointers to the hardware peripherals.
 * The cast from integer to pointer tells the compiler "this address is
 * where the registers live" — reads/writes go straight to hardware. */
static struct I2C_T   *i2c   = (struct I2C_T   *)AV_CONFIG_BASE;
static struct AUDIO_T *audio = (struct AUDIO_T *)AUDIO_BASE;

/*
 * i2c_wait — spin until the current I2C byte transfer finishes.
 *
 * After writing a command to cmd_status, the hardware starts clocking
 * bits out onto the SDA/SCL wires.  The TIP (Transfer In Progress) bit
 * stays high until the last bit has been sent and the device has had a
 * chance to ACK.  We must wait here before issuing the next byte.
 */
static void i2c_wait(void)
{
    while (i2c->cmd_status & I2C_TIP);  /* spin until TIP clears */
}

/*
 * i2c_write_byte — load one byte into the I2C controller and start sending it.
 *
 * Parameters:
 *   data — the byte to put on the bus
 *   cmd  — command flags (I2C_START, I2C_WRITE, I2C_STOP — can be OR'd)
 *
 * The hardware sends the byte and generates the START/STOP conditions
 * as requested.  We wait for TIP to clear before returning.
 */
static void i2c_write_byte(unsigned char data, unsigned char cmd)
{
    i2c->data       = data;   /* load the byte to send     */
    i2c->cmd_status = cmd;    /* pull the trigger           */
    i2c_wait();               /* wait until hardware is done */
}

/*
 * wm8731_write — write one register of the WM8731 audio codec.
 *
 * The WM8731's I2C protocol uses 2-byte transfers:
 *   Byte 0: bits [7:1] = 6-bit register address, bit [0] = data bit 8
 *   Byte 1: bits [7:0] = data bits [7:0]
 *
 * So a 9-bit value is split across both bytes.  The address is packed
 * into the top 7 bits of the first byte.
 *
 * Full transaction on the wire:
 *   START → 0x34 (WM8731 address + write) → byte0 → byte1 → STOP
 *
 * Parameters:
 *   reg — WM8731 register number (0–15)
 *   val — 9-bit value to write into that register
 */
static void wm8731_write(unsigned char reg, unsigned short val)
{
    /* Pack register address into upper 7 bits; MSB of val into bit 0 */
    unsigned char b0 = (reg << 1) | ((val >> 8) & 0x01);
    unsigned char b1 = val & 0xFF;   /* lower 8 bits of val */

    /* Send the codec's I2C address with a START condition */
    i2c_write_byte((WM8731_ADDR << 1), I2C_START);  /* 0x34 = 0x1A<<1 | 0 (write) */
    /* Send first data byte (no start, no stop yet) */
    i2c_write_byte(b0, I2C_WRITE);
    /* Send second data byte and close the transaction with STOP */
    i2c_write_byte(b1, I2C_WRITE | I2C_STOP);
}

/*
 * init_audio_codec — configure the WM8731 for our use.
 *
 * This must run once before any audio is sampled.  It programs the codec
 * over I2C to:
 *   - Accept microphone input (not line-in)
 *   - Output audio to the headphone/line-out jack
 *   - Sample at 8 kHz (to match the training pipeline)
 *   - Use 16-bit I2S serial format
 *   - Keep everything powered on
 *
 * Why 8 kHz?  Human speech intelligibility is preserved below 4 kHz
 * (Nyquist: you need 2× the highest frequency you care about).  8 kHz
 * also means our 2-second window is only 16000 samples — small enough
 * to fit in a few hundred KB of SDRAM.
 *
 * Register map (see WM8731 datasheet for full details):
 *   R0  — Left  line input volume, unmuted (0x017)
 *   R1  — Right line input volume, unmuted (0x017)
 *   R4  — Analogue path: MIC selected (INSEL=1), DAC on (DACSEL=1) (0x014)
 *   R5  — Digital path: no high-pass filter (0x000)
 *   R6  — Power management: all blocks on (0x000)
 *   R7  — Digital audio interface: I2S format, 16-bit (0x00A)
 *   R8  — Sample rate: 8 kHz, USB clock mode (0x00D)
 *   R9  — Active control: start the codec (0x001)
 *   R15 — Reset register: writing any value resets to defaults (done first)
 *
 * Note on R4 = 0x014:
 *   Bit 2 (INSEL)  = 1 → microphone input selected
 *   Bit 4 (DACSEL) = 1 → DAC output enabled (needed for passthrough)
 *   Bit 1 (MUTEMIC)= 0 → mic is NOT muted
 *   Bit 0 (MICBOOST)=0 → no +20 dB boost (add if mic is too quiet)
 *   If you are using the blue LINE-IN jack instead, change to 0x010
 *   (INSEL=0, DACSEL=1).
 */
static void init_audio_codec(void)
{
    /* Configure the I2C controller clock.
     * We need SCL to run at 400 kHz (fast-mode I2C).
     * System clock = 50 MHz.
     * Prescaler formula: prescale = (50e6 / (5 × 400e3)) − 1 = 24.
     * The controller divides the system clock by (prescale+1)×5. */
    i2c->prescale_lo = 24;   /* low  byte of prescaler = 24 */
    i2c->prescale_hi = 0;    /* high byte of prescaler =  0 (fits in 8 bits) */
    i2c->control     = 0x80; /* bit 7 = 1: enable the I2C core */

    /* Reset all WM8731 registers to factory defaults before configuring */
    wm8731_write(15, 0x000);

    /* R0, R1: set both input channels to 0 dB gain, unmuted */
    wm8731_write(0, 0x017);   /* left  line in: 0 dB, unmuted */
    wm8731_write(1, 0x017);   /* right line in: 0 dB, unmuted */

    /* R4: select microphone as ADC input, enable DAC output path */
    wm8731_write(4, 0x014);

    /* R5: disable digital high-pass filter (we want raw samples) */
    wm8731_write(5, 0x000);

    /* R6: power down register — writing 0x000 keeps EVERYTHING powered on
     * (each bit in this register powers DOWN a block when set to 1) */
    wm8731_write(6, 0x000);

    /* R7: I2S audio format, 16-bit word length */
    wm8731_write(7, 0x00A);

    /* R8: 8 kHz sample rate using USB clock mode
     * (USB/Normal=1, BOSR=0, SR=0011, CLKIDIV2=0, CLKODIV2=0) */
    wm8731_write(8, 0x00D);

    /* R9: activate the codec — it will now start sampling */
    wm8731_write(9, 0x001);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * CIRCULAR (RING) BUFFER
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * A circular buffer is an array treated as if the end wraps back to the
 * beginning.  We use it to keep a rolling window of the most recent audio.
 *
 * Picture it as a clock face with 40000 positions:
 *
 *   circ_write always points to the NEXT position to write.
 *   After each write: circ_write = (circ_write + 1) % CIRC_BUF_SIZE
 *   This wraps circ_write back to 0 after it reaches 39999.
 *
 * At any point in time, the buffer contains the last 5 seconds of audio,
 * starting at circ_write (oldest) and ending at circ_write-1 (newest).
 *
 * Why not just a plain array with an index?
 *   Because we would have to shift 40000 floats on every new sample to keep
 *   the newest at the end.  The circular approach does it in one write + one
 *   modulo — constant time regardless of buffer size.
 *
 * Buffer size:
 *   CIRC_BUF_SIZE = 40000 = 5 seconds × 8000 samples/second.
 *   This matches the length of your phone recordings so the KEY[0] audio
 *   dump captures exactly one full 5-second clip.
 *
 *   IMPORTANT: the training script (train_voice_gate.py) clips every audio
 *   file to TARGET_LENGTH = 16000 samples (2 seconds) before extracting
 *   MFCCs.  This means only the FIRST 2 seconds of each dump are used for
 *   training.  Say "open sesame" within the first 2 seconds of the window,
 *   or manually trim the WAV files before training.
 *
 * AUDIO_WINDOW_LEN = 15880 is the exact number of samples the MFCC function
 * needs.  Derived from: N_FFT + HOP_LENGTH × (N_FRAMES − 1)
 *                      = 200   + 160       × (99       − 1)
 *                      = 200   + 15680 = 15880 samples ≈ 1.985 seconds.
 */
#define CIRC_BUF_SIZE    40000   /* 5 s of audio at 8 kHz — matches phone recordings */
#define INFERENCE_STRIDE 24000   /* run inference every 3 s of new audio             */
#define AUDIO_WINDOW_LEN 15880   /* samples fed to MFCC (fixed by model architecture) */

/* The circular buffer itself.  Static global → lives in SDRAM, not stack.
 * 40000 floats × 4 bytes = 160 KB. */
static float circ_buf[CIRC_BUF_SIZE];

/* Index of the next write position in circ_buf (wraps 0 → 39999 → 0 → …) */
static int circ_write  = 0;

/* Counts new samples since the last inference run.
 * When this reaches INFERENCE_STRIDE (24000), we trigger MFCC + CNN. */
static int new_samples = 0;

/*
 * Working buffers for the inference pipeline.  All declared static so they
 * live in SDRAM.  Putting them on the stack would overflow it immediately
 * (the stack is ~8 KB; these arrays need hundreds of KB).
 *
 * mfcc_buf     [13][99] floats — output of compute_mfcc(), input to run_inference()
 * audio_window [15880]  floats — flat slice from circ_buf, passed to compute_mfcc()
 * prob         [2]      floats — CNN output: prob[0]=P(not open sesame), prob[1]=P(open sesame)
 */
static float mfcc_buf[N_MFCC][N_FRAMES];
static float audio_window[AUDIO_WINDOW_LEN];
static float prob[2];


/* ═══════════════════════════════════════════════════════════════════════════
 * GATE CONTROL
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The motor is not yet wired.  open_and_close_gate() is defined here but
 * NOT called during normal operation — the gate trigger only prints
 * "TRIGGERED" to the terminal and VGA.
 *
 * When the motor is wired:
 *   1. Connect the LEGO motor controller to the JP1 GPIO header.
 *   2. Un-comment the open_and_close_gate() call in the prob[1] > 0.90 branch.
 *   3. Tune OPEN_DELAY and HOLD_DELAY until the gate moves the right amount.
 *
 * OPEN_DELAY = 50000000: the number of loop iterations delay() counts to.
 * At ~100 MHz Nios V, one iteration ≈ 10 ns, so 50M iterations ≈ 0.5 s.
 * Increase this value if the gate doesn't open fully.
 *
 * HOLD_DELAY = 20000000: how long the gate stays open (~0.2 s at 100 MHz).
 */
#define OPEN_DELAY  50000000
#define HOLD_DELAY  20000000

/*
 * open_and_close_gate — runs the motor sequence to open then close the gate.
 *
 * Sequence:
 *   1. Spin motor 0 clockwise     for OPEN_DELAY  cycles → gate opens
 *   2. Stop motor                 for HOLD_DELAY  cycles → gate held open
 *   3. Spin motor 0 counter-CW   for OPEN_DELAY  cycles → gate closes
 *   4. Stop motor
 *
 * The motor number (0) refers to the first motor port on the controller.
 */
static void open_and_close_gate(void)
{
    spin_motor(0, 1);       /* 1 = clockwise → open gate      */
    delay(OPEN_DELAY);
    stop_motor(0);
    delay(HOLD_DELAY);      /* hold open so a person can walk through */
    spin_motor(0, 0);       /* 0 = counter-clockwise → close gate    */
    delay(OPEN_DELAY);
    stop_motor(0);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * AUDIO DUMP  (triggered by KEY[0])
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Purpose:
 *   The model was trained on phone recordings but the board uses a lab mic.
 *   This mismatch causes false positives (confidence 1.00 on silence).
 *   The fix is to re-record training samples using the SAME mic and hardware
 *   pipeline.  This dump feature lets you do that.
 *
 * How it works end-to-end:
 *
 *   ON THE BOARD:
 *     1. The circular buffer always holds the last 5 seconds of audio.
 *     2. You say "open sesame" (or any other word for negative samples).
 *     3. You press KEY[0] on the DE1-SoC (the leftmost black pushbutton).
 *     4. The board prints "DUMP_START" to the terminal, then prints all
 *        40000 samples as signed decimal integers (one per line), then
 *        prints "DUMP_END".
 *
 *   ON THE PC:
 *     5. You have already redirected nios2-terminal output to a file:
 *          nios2-terminal.exe --instance 0 | Tee-Object -FilePath capture.txt
 *     6. After DUMP_END appears, Ctrl+C the terminal.
 *     7. Run:
 *          python training/capture_wav.py capture.txt my_sample.wav
 *        This script finds the DUMP_START/DUMP_END block, parses the integers,
 *        and writes a standard 8 kHz 16-bit mono WAV file.
 *     8. Move the WAV to training/dataset/positive/ or negative/.
 *     9. Re-run train_voice_gate.py, then export_weights.py, then recompile.
 *
 * Why integers, not floats?
 *   The samples in circ_buf are normalised floats in [-1.0, 1.0].
 *   Printing a full float ("0.00123456") over JTAG UART takes ~12 chars each.
 *   Converting to a 16-bit integer (range -32767 to 32767) and printing as
 *   a decimal number takes only ~5 chars on average — about 2.5× smaller,
 *   so the dump finishes faster.
 *   capture_wav.py reverses the conversion when it writes the WAV file.
 *
 * KEY[0] is active-low:
 *   Reading the KEY register gives 1 when the button is NOT pressed,
 *   and 0 when it IS pressed.  We detect a fresh press by watching for
 *   the transition from 1 → 0 (key_prev == 1, key_now == 0).
 *   Without this edge-detection the dump would fire repeatedly for every
 *   loop iteration while the button is held down.
 *
 * WARNING: the dump takes several seconds (40000 samples × ~7 chars = ~280 KB
 * over ~50 KB/s USB Blaster).  The audio FIFO will overflow during this time
 * (samples arriving at 8 kHz, nobody draining them).  Audio captured AFTER
 * the dump starts will contain artifacts.  That is fine — just press the button
 * AFTER speaking, not before.
 */

/* Pointer to the KEY pushbutton register (4 buttons, active-low, bits [3:0]) */
static volatile unsigned int *keys = (volatile unsigned int *)KEY_BASE;

/*
 * dump_audio_window — serialise circ_buf to JTAG UART as ASCII integers.
 *
 * Walks the circular buffer from oldest sample (circ_write) to newest
 * (circ_write - 1, wrapping around), converting each float to a 16-bit
 * integer and printing it.
 *
 * The output format is:
 *   DUMP_START\r\n
 *   -1243\r\n
 *   8821\r\n
 *   0\r\n
 *   … (40000 lines total) …
 *   DUMP_END\r\n
 */
static void dump_audio_window(void)
{
    int i;
    jtag_puts("\r\nDUMP_START\r\n");

    for (i = 0; i < CIRC_BUF_SIZE; i++) {
        /* Read samples in order from OLDEST to NEWEST.
         * circ_write points to the next write slot, so circ_write is the
         * OLDEST sample still in the buffer.
         * (circ_write + i) % CIRC_BUF_SIZE walks forward, wrapping as needed. */
        int sample = (int)(circ_buf[(circ_write + i) % CIRC_BUF_SIZE] * 32767.0f);
        jtag_put_int(sample);
        jtag_puts("\r\n");
    }

    jtag_puts("DUMP_END\r\n");
}


/* ═══════════════════════════════════════════════════════════════════════════
 * DIAGNOSTIC LED MAP
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The 10 red LEDs (LEDR[9:0]) show the pipeline state at a glance.
 * If the program hangs, the last LED that stays lit shows where it stopped.
 *
 *  LED[0]  always ON         — boot complete, codec initialised
 *  LED[1]  pulses ~10× / s   — audio samples are flowing from the ADC FIFO
 *  LED[2]  ON briefly         — 3 s of audio accumulated, inference starting
 *  LED[3]  ON briefly         — MFCC extraction finished
 *  LED[4]  ON                 — CNN inference finished
 *  LED[5]  ON  if prob > 0.50 — weak match (model leaning toward "open sesame")
 *  LED[6]  ON  if prob > 0.70 — moderate confidence
 *  LED[7]  ON  if prob > 0.80 — strong confidence
 *  LED[8]  ON  if prob > 0.90 — threshold met — would trigger gate
 *  LED[9]  ON  if prob > 0.95 — very high confidence
 *
 * After each inference cycle all LEDs except LED[0] are cleared (idle state).
 *
 * LED values are written as a bitmask to the LEDR register at LED_BASE:
 *   0x001 = 0000000001 = only LED[0]
 *   0x005 = 0000000101 = LED[0] + LED[2]
 *   0x009 = 0000001001 = LED[0] + LED[3]
 *   0x011 = 0000010001 = LED[0] + LED[4]
 * ═══════════════════════════════════════════════════════════════════════════ */


/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN — entry point
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    int i;

    /* key_prev tracks the last known state of KEY[0] so we can detect
     * a falling edge (released→pressed transition).
     * Start at 1 (released) because the button is active-low. */
    int key_prev = 1;

    /* ── Initialisation ───────────────────────────────────────────────── */

    /* Set up GPIO pins as outputs and ensure all motors are stopped.
     * Must run before anything else so the motor pins are in a safe state. */
    setup_gpio();
    stop_all_motors();

    /* Configure the WM8731 codec over I2C.
     * After this call the codec is sampling the mic at 8 kHz and filling
     * the ADC FIFO with 16-bit samples. */
    init_audio_codec();

    /* Flush stale ADC samples.
     * During codec initialisation, the hardware may have put garbage values
     * into the ADC FIFO (the codec was settling into its new configuration).
     * We drain those here by reading and discarding everything currently
     * queued, so the first samples we process are clean.
     * Without this, those garbage samples get passed through to the speakers
     * as a pop or burst of noise at startup. */
    while (audio->rarc) { (void)audio->ldata; (void)audio->rdata; }

    /* Get a pointer to the LED register so we can update LEDs with a single write */
    volatile unsigned int *leds = (volatile unsigned int *)LED_BASE;

    /* Light LED[0] to signal "boot complete, codec ready, entering main loop" */
    *leds = 0x001;

    /* ── Boot messages ────────────────────────────────────────────────── */

    /* Clear the VGA screen and write the static title and initial status */
    vga_clear();
    vga_status(0, "=== Open Sesame Gate ===");  /* row 0 — permanent title */
    vga_status(1, "LISTENING...");               /* row 1 — live status     */

    /* Print the same information to the JTAG UART terminal */
    jtag_puts("\r\n=== Open Sesame Gate ===\r\n");
    jtag_puts("Audio codec initialised. LISTENING...\r\n");

    /* ═══════════════════════════════════════════════════════════════════
     * MAIN LOOP
     * ═══════════════════════════════════════════════════════════════════
     *
     * This loop runs forever.  It does two things on every iteration:
     *
     *   A. Check KEY[0] — if just pressed, dump the audio buffer.
     *   B. Check the ADC FIFO — if a sample is ready, process it.
     *
     * The loop is NOT interrupt-driven.  It polls both peripherals every
     * iteration.  This is called "polling" or "busy-waiting".  It works
     * fine here because:
     *   - The ADC produces samples at 8 kHz = one every 125 µs.
     *   - The Nios V at ~100 MHz can execute thousands of instructions
     *     in 125 µs, so it will never miss a sample during normal operation.
     *   - During MFCC + CNN (which takes several seconds), the ADC FIFO
     *     overflows and audio is lost.  This is expected — we resume
     *     listening from a fresh window after inference completes.
     * ═══════════════════════════════════════════════════════════════════ */
    while (1) {

        /* ── A. KEY[0] edge detection ─────────────────────────────── */

        /* Read the current state of KEY[0].
         * The KEY register holds all 4 buttons in bits [3:0].
         * Bit 0 = KEY[0].  Active-low: 0 = pressed, 1 = released. */
        int key_now = (*keys & 0x1);

        /* Detect a falling edge: was released (1), now pressed (0) */
        if (key_now == 0 && key_prev == 1) {
            /* The button was just pressed — dump the audio buffer */
            jtag_puts("\r\nDumping audio buffer...\r\n");
            dump_audio_window();
            jtag_puts("Done. Run: python training/capture_wav.py capture.txt out.wav\r\n");
        }
        key_prev = key_now;  /* remember state for next iteration */

        /* ── B. Audio FIFO polling ────────────────────────────────── */

        /* audio->rarc is non-zero when the ADC FIFO has at least one sample.
         * We use 'if' not 'while' so that KEY[0] is checked on every sample
         * and the loop doesn't get stuck draining a rapidly-filling FIFO
         * without ever servicing the key. */
        if (audio->rarc) {

            /* Pop one 16-bit sample from the LEFT channel ADC FIFO.
             * The register is 32 bits wide; the sample is in bits [15:0].
             * We must also read rdata to advance the FIFO read pointer,
             * even though we don't use the right-channel value. */
            int raw_left = audio->ldata;
            (void)audio->rdata;   /* discard right channel, but still drain it */

            /* ── Software audio passthrough ───────────────────────────
             * Write the raw sample back to the DAC FIFOs so you can hear
             * the mic through the speakers.  This is ONLY for debugging —
             * it lets you confirm the mic is picking up sound.
             *
             * We check wsrc/wslc (write space) first.  If the DAC FIFO is
             * full (e.g. during the long MFCC/CNN computation when we stop
             * draining the ADC), we skip the write rather than stalling.
             * Skipping a few DAC samples causes a brief glitch in the
             * playback, but that is better than blocking the whole loop. */
            if (audio->wsrc) audio->ldata = raw_left;  /* left  speaker */
            if (audio->wslc) audio->rdata = raw_left;  /* right speaker */

            /* ── Normalise and store ───────────────────────────────────
             * Convert the raw 16-bit signed integer to a float in [-1, 1].
             * (short)(raw_left & 0xFFFF) extracts the lower 16 bits and
             * interprets them as a signed 16-bit integer (two's complement).
             * Dividing by 32768.0 scales to the [-1, 1] range that the
             * MFCC function expects. */
            float sample = (float)(short)(raw_left & 0xFFFF) / 32768.0f;

            /* Store in the circular buffer at the current write position,
             * then advance the write pointer with wrap-around. */
            circ_buf[circ_write] = sample;
            circ_write = (circ_write + 1) % CIRC_BUF_SIZE;
            new_samples++;

            /* ── LED[1] heartbeat ─────────────────────────────────────
             * Toggle LED[1] every 800 samples (~0.1 s) to show that audio
             * is flowing.  0x31F = 0b 0011 0001 1111 = 799 in binary.
             * (new_samples & 0x31F) == 0 is true every 800 samples.
             *
             * We use XOR (^=) with mask 0x002 (bit 1) to toggle just LED[1]
             * without disturbing LED[0] which is always on. */
            if ((new_samples & 0x31F) == 0)
                *leds ^= 0x002;

            /* ── Inference trigger ────────────────────────────────────
             * Once we have accumulated INFERENCE_STRIDE = 24000 new samples
             * (3 seconds of audio), run the full MFCC + CNN pipeline. */
            if (new_samples >= INFERENCE_STRIDE) {
                new_samples = 0;  /* reset counter for the next window */

                /* LED[2] on = "inference window ready, about to process" */
                *leds = 0x005;
                jtag_puts("\r\nPROCESSING...\r\n");
                vga_status(1, "PROCESSING...");

                /* ── Build flat audio window ──────────────────────────
                 * The MFCC function wants a contiguous array of
                 * AUDIO_WINDOW_LEN (15880) samples.  We read the last
                 * 15880 samples from the circular buffer in chronological
                 * order (oldest first).
                 *
                 * 'start' is the index of the oldest of those 15880 samples.
                 * It is computed by going back 15880 positions from the
                 * current write pointer, wrapping around if needed:
                 *   start = (circ_write - 15880 + 40000) % 40000
                 * The +40000 prevents the modulo from operating on a
                 * negative number (C's % can return negative for negatives). */
                int start = (circ_write - AUDIO_WINDOW_LEN + CIRC_BUF_SIZE)
                            % CIRC_BUF_SIZE;

                for (i = 0; i < AUDIO_WINDOW_LEN; i++)
                    audio_window[i] = circ_buf[(start + i) % CIRC_BUF_SIZE];

                /* ── MFCC extraction ──────────────────────────────────
                 * compute_mfcc() transforms the 15880-sample audio window
                 * into a 13×99 matrix of MFCC features.
                 * Each column is one time frame (~20 ms of audio).
                 * Each row is one MFCC coefficient (a spectral shape feature).
                 * The resulting matrix describes HOW the sound's spectrum
                 * changes over time — this is what the CNN was trained on. */
                compute_mfcc(audio_window, mfcc_buf);

                /* LED[3] on = MFCC done */
                *leds = 0x009;
                jtag_puts("  MFCC extraction done\r\n");
                vga_status(1, "MFCC done, running CNN...");

                /* ── CNN inference ────────────────────────────────────
                 * run_inference() runs the trained neural network on the
                 * 13×99 MFCC feature matrix and fills prob[2]:
                 *   prob[0] = probability that this is NOT "open sesame"
                 *   prob[1] = probability that this IS  "open sesame"
                 * prob[0] + prob[1] == 1.0 (softmax output). */
                run_inference((const float (*)[N_FRAMES])mfcc_buf, prob);

                /* LED[4] on = CNN done, about to show results */
                *leds = 0x011;
                jtag_puts("  CNN inference done\r\n");
                jtag_puts("  Confidence: ");
                jtag_put_float(prob[1]);   /* prob[1] = P("open sesame") */
                jtag_puts("\r\n");

                /* Update VGA row 1 with the confidence value.
                 * "Confidence: " is 12 characters, so the number starts at col 12. */
                vga_status(1, "Confidence: ");
                vga_put_float_at(1, 12, prob[1]);

                /* ── Confidence LED bar graph ─────────────────────────
                 * LEDs 5–9 light up progressively as confidence rises.
                 * This gives a visual "meter" showing how close you are to
                 * the threshold, even before the terminal output appears. */
                unsigned int conf_leds = 0x011;   /* start with LED[0]+LED[4] */
                if (prob[1] > 0.50f) conf_leds |= (1 << 5);  /* weak match    */
                if (prob[1] > 0.70f) conf_leds |= (1 << 6);  /* moderate      */
                if (prob[1] > 0.80f) conf_leds |= (1 << 7);  /* strong        */
                if (prob[1] > 0.90f) conf_leds |= (1 << 8);  /* threshold met */
                if (prob[1] > 0.95f) conf_leds |= (1 << 9);  /* very high     */
                *leds = conf_leds;

                /* ── Gate trigger ─────────────────────────────────────
                 * If confidence exceeds the 0.90 threshold, print TRIGGERED.
                 * open_and_close_gate() is intentionally NOT called here
                 * because the motor is not yet wired.  When it is wired,
                 * un-comment that call. */
                if (prob[1] > 0.90f) {
                    jtag_puts("  *** TRIGGERED! ***\r\n");
                    vga_status(1, "*** TRIGGERED! ***");
                    /* open_and_close_gate(); ← un-comment when motor is wired */
                }

                /* Hold for ~0.3 s at the current LED state so you have time
                 * to read the confidence LEDs before they reset.
                 * delay(3000000) ≈ 30 ms at 100 MHz — long enough to see,
                 * short enough not to miss the next speech window. */
                delay(3000000);

                /* Reset LEDs to idle (only LED[0] on) and print LISTENING again */
                *leds = 0x001;
                jtag_puts("LISTENING...\r\n");
                vga_status(1, "LISTENING...");
            }
        }
    }

    return 0;   /* never reached — main loop runs forever */
}

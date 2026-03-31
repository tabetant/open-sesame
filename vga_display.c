/*
 * vga_display.c — VGA pixel graphics + WM8731 audio for open-sesame gate.
 *
 * Follows ECE243 part3.c conventions exactly:
 *   - Buffer1/Buffer2 static arrays, pixel_buffer_start global
 *   - wait_for_vsync() polling pattern
 *   - draw_pixel() via (pixel_buffer_start + (y<<10) + (x<<1))
 *   - Bresenham draw_line()
 *
 * STATE 1 shows a countdown bar that drains over the 2-second audio window.
 * Call vga_update_listening(new_samples, INFERENCE_STRIDE) every ~800 samples
 * from main.c to animate it.
 *
 * Audio: square-wave tones written directly to WM8731 ldata/rdata.
 * No malloc, no printf, no stdlib beyond the nothing-included here.
 */

#include "vga_display.h"
#include "address_map.h"

/* ── Double-buffer setup (ECE243 part3.c style) ─────────────────────────── */
int pixel_buffer_start;   /* points to the back buffer being drawn into     */

/* 240 rows × 512 shorts — row stride in hardware is 1024 bytes = 512 shorts */
static short int Buffer1[240][512];
static short int Buffer2[240][512];

/* ── Hardware register pointers ─────────────────────────────────────────── */
static volatile int *pixel_ctrl_ptr = (volatile int *)0xFF203020;

struct AUDIO_REG {
    volatile unsigned int  control;
    volatile unsigned char rarc;
    volatile unsigned char ralc;
    volatile unsigned char wsrc;
    volatile unsigned char wslc;
    volatile int ldata;
    volatile int rdata;
};
static volatile struct AUDIO_REG *audio_hw = (volatile struct AUDIO_REG *)0xFF203040;

/* ── 5×7 bitmap font ────────────────────────────────────────────────────── */
static const unsigned char font5x7[][5] = {
    /* space */ {0x00,0x00,0x00,0x00,0x00},
    /* !     */ {0x00,0x00,0x5F,0x00,0x00},
    /* "     */ {0x00,0x07,0x00,0x07,0x00},
    /* #     */ {0x14,0x7F,0x14,0x7F,0x14},
    /* $     */ {0x24,0x2A,0x7F,0x2A,0x12},
    /* %     */ {0x23,0x13,0x08,0x64,0x62},
    /* &     */ {0x36,0x49,0x55,0x22,0x50},
    /* '     */ {0x00,0x05,0x03,0x00,0x00},
    /* (     */ {0x00,0x1C,0x22,0x41,0x00},
    /* )     */ {0x00,0x41,0x22,0x1C,0x00},
    /* *     */ {0x08,0x2A,0x1C,0x2A,0x08},
    /* +     */ {0x08,0x08,0x3E,0x08,0x08},
    /* ,     */ {0x00,0x50,0x30,0x00,0x00},
    /* -     */ {0x08,0x08,0x08,0x08,0x08},
    /* .     */ {0x00,0x60,0x60,0x00,0x00},
    /* /     */ {0x20,0x10,0x08,0x04,0x02},
    /* 0     */ {0x3E,0x51,0x49,0x45,0x3E},
    /* 1     */ {0x00,0x42,0x7F,0x40,0x00},
    /* 2     */ {0x42,0x61,0x51,0x49,0x46},
    /* 3     */ {0x21,0x41,0x45,0x4B,0x31},
    /* 4     */ {0x18,0x14,0x12,0x7F,0x10},
    /* 5     */ {0x27,0x45,0x45,0x45,0x39},
    /* 6     */ {0x3C,0x4A,0x49,0x49,0x30},
    /* 7     */ {0x01,0x71,0x09,0x05,0x03},
    /* 8     */ {0x36,0x49,0x49,0x49,0x36},
    /* 9     */ {0x06,0x49,0x49,0x29,0x1E},
    /* :     */ {0x00,0x36,0x36,0x00,0x00},
    /* ;     */ {0x00,0x56,0x36,0x00,0x00},
    /* <     */ {0x00,0x08,0x14,0x22,0x41},
    /* =     */ {0x14,0x14,0x14,0x14,0x14},
    /* >     */ {0x41,0x22,0x14,0x08,0x00},
    /* ?     */ {0x02,0x01,0x51,0x09,0x06},
    /* @     */ {0x32,0x49,0x79,0x41,0x3E},
    /* A     */ {0x7E,0x11,0x11,0x11,0x7E},
    /* B     */ {0x7F,0x49,0x49,0x49,0x36},
    /* C     */ {0x3E,0x41,0x41,0x41,0x22},
    /* D     */ {0x7F,0x41,0x41,0x22,0x1C},
    /* E     */ {0x7F,0x49,0x49,0x49,0x41},
    /* F     */ {0x7F,0x09,0x09,0x09,0x01},
    /* G     */ {0x3E,0x41,0x41,0x49,0x7A},
    /* H     */ {0x7F,0x08,0x08,0x08,0x7F},
    /* I     */ {0x00,0x41,0x7F,0x41,0x00},
    /* J     */ {0x20,0x40,0x41,0x3F,0x01},
    /* K     */ {0x7F,0x08,0x14,0x22,0x41},
    /* L     */ {0x7F,0x40,0x40,0x40,0x40},
    /* M     */ {0x7F,0x02,0x04,0x02,0x7F},
    /* N     */ {0x7F,0x04,0x08,0x10,0x7F},
    /* O     */ {0x3E,0x41,0x41,0x41,0x3E},
    /* P     */ {0x7F,0x09,0x09,0x09,0x06},
    /* Q     */ {0x3E,0x41,0x51,0x21,0x5E},
    /* R     */ {0x7F,0x09,0x19,0x29,0x46},
    /* S     */ {0x46,0x49,0x49,0x49,0x31},
    /* T     */ {0x01,0x01,0x7F,0x01,0x01},
    /* U     */ {0x3F,0x40,0x40,0x40,0x3F},
    /* V     */ {0x1F,0x20,0x40,0x20,0x1F},
    /* W     */ {0x3F,0x40,0x38,0x40,0x3F},
    /* X     */ {0x63,0x14,0x08,0x14,0x63},
    /* Y     */ {0x03,0x04,0x78,0x04,0x03},
    /* Z     */ {0x61,0x51,0x49,0x45,0x43},
    /* [     */ {0x00,0x00,0x7F,0x41,0x41},
    /* \     */ {0x02,0x04,0x08,0x10,0x20},
    /* ]     */ {0x41,0x41,0x7F,0x00,0x00},
    /* ^     */ {0x04,0x02,0x01,0x02,0x04},
    /* _     */ {0x40,0x40,0x40,0x40,0x40},
};

/* ── Low-level draw helpers ─────────────────────────────────────────────── */

static void wait_for_vsync(void)
{
    int status;
    *pixel_ctrl_ptr = 1;
    status = *(pixel_ctrl_ptr + 3);
    while ((status & 0x01) != 0)
        status = *(pixel_ctrl_ptr + 3);
}

static void draw_pixel(int x, int y, short int color)
{
    volatile short int *addr =
        (volatile short int *)(pixel_buffer_start + (y << 10) + (x << 1));
    *addr = color;
}

static void clear_screen(short int color)
{
    int x, y;
    for (y = 0; y < SCREEN_H; y++)
        for (x = 0; x < SCREEN_W; x++)
            draw_pixel(x, y, color);
}

/* Bresenham line — identical to ECE243 part3.c */
static void draw_line(int x0, int y0, int x1, int y1, short int color)
{
    int dx = x1 - x0;
    int dy = y1 - y0;
    int sx = (dx > 0) ? 1 : -1;
    int sy = (dy > 0) ? 1 : -1;
    int x = x0, y = y0, i;

    dx = (dx < 0) ? -dx : dx;
    dy = (dy < 0) ? -dy : dy;

    if (dx >= dy) {
        int e = 2 * dy - dx;
        for (i = 0; i <= dx; i++) {
            draw_pixel(x, y, color);
            if (e > 0) { y += sy; e -= 2 * dx; }
            e += 2 * dy;
            x += sx;
        }
    } else {
        int e = 2 * dx - dy;
        for (i = 0; i <= dy; i++) {
            draw_pixel(x, y, color);
            if (e > 0) { x += sx; e -= 2 * dy; }
            e += 2 * dx;
            y += sy;
        }
    }
}

static void fill_rect(int x, int y, int w, int h, short int color)
{
    int px, py;
    for (py = y; py < y + h; py++)
        for (px = x; px < x + w; px++)
            draw_pixel(px, py, color);
}

/* ── Text rendering ─────────────────────────────────────────────────────── */

static void draw_char(int x, int y, char c, short int color, int scale)
{
    int col, row, sx, sy, idx;
    if (c < ' ' || c > '_') c = ' ';
    idx = (int)(c - ' ');
    for (col = 0; col < 5; col++) {
        unsigned char bits = font5x7[idx][col];
        for (row = 0; row < 7; row++) {
            if (bits & (1 << row)) {
                for (sy = 0; sy < scale; sy++)
                    for (sx = 0; sx < scale; sx++)
                        draw_pixel(x + col*scale + sx,
                                   y + row*scale + sy,
                                   color);
            }
        }
    }
}

static int str_len(const char *s)
{
    int n = 0; while (*s++) n++; return n;
}

static void draw_string(int x, int y, const char *s, short int color, int scale)
{
    int char_w = 6 * scale;
    while (*s) { draw_char(x, y, *s, color, scale); x += char_w; s++; }
}

static void draw_string_centered(int y, const char *s, short int color, int scale)
{
    int x = (SCREEN_W - str_len(s) * 6 * scale) / 2;
    draw_string(x, y, s, color, scale);
}

/* ── Audio — square-wave tone synthesis ─────────────────────────────────── */

static void play_tone(int freq_hz, int duration_ms)
{
    int total  = (8000 * duration_ms) / 1000;
    int half   = (freq_hz > 0) ? (8000 / (2 * freq_hz)) : 1;
    int sample, phase = 0;

    for (sample = 0; sample < total; sample++) {
        int val = (phase < half) ? 0x007FFFFF : (int)0xFF800001;
        while (!audio_hw->wsrc);
        audio_hw->ldata = val;
        audio_hw->rdata = val;
        phase++;
        if (phase >= 2 * half) phase = 0;
    }
}

/* ── Processing state — thinking words + dot animation ─────────────────── */

static const char *thinking_words[] = {
    "COGITATING...",
    "DELIBERATING...",
    "FLIBBERJIBBETING...",
    "HMMMM...",
    "PROCESSING...",
    "ONE MOMENT...",
    "CONSULTING THE ORACLE...",
    "THINKING REALLY HARD...",
    "ALMOST THERE...",
    "NEARLY..."
};
#define NUM_THINKING_WORDS  10

static int thinking_index = 0;
static const int proc_freqs[3] = { 440, 523, 659 };

/* ── Busy-wait delay ────────────────────────────────────────────────────── */

static void busy_wait(int count)
{
    volatile int c = count; while (c--);
}

/* ── Countdown bar helpers ──────────────────────────────────────────────── */
/*
 * The bar sits at the bottom third of the screen.
 * It starts FULL GREEN when the 2-second window opens and drains to empty
 * as samples are collected.  The outer outline is always white.
 */
#define BAR_X   30
#define BAR_Y   168
#define BAR_W   260
#define BAR_H   22

/* Draw the static listening screen (text only, no bar fill yet).
 * Used as the base for both vga_state_listening and vga_update_listening. */
static void draw_listening_base(void)
{
    clear_screen(COLOR_BLACK);
    draw_string_centered(55,  "YOU HAVE REACHED THE GATE", COLOR_WHITE,  2);
    draw_string_centered(90,  "SPEAK THE PASSWORD",        COLOR_YELLOW, 2);
    draw_string_centered(140, "SPEAK NOW",                 COLOR_CYAN,   2);
    draw_string_centered(156, "2 SECOND WINDOW",           COLOR_CYAN,   1);

    /* Bar outline */
    draw_line(BAR_X - 1,         BAR_Y - 1,         BAR_X + BAR_W,     BAR_Y - 1,         COLOR_WHITE);
    draw_line(BAR_X - 1,         BAR_Y + BAR_H,     BAR_X + BAR_W,     BAR_Y + BAR_H,     COLOR_WHITE);
    draw_line(BAR_X - 1,         BAR_Y - 1,         BAR_X - 1,         BAR_Y + BAR_H,     COLOR_WHITE);
    draw_line(BAR_X + BAR_W,     BAR_Y - 1,         BAR_X + BAR_W,     BAR_Y + BAR_H,     COLOR_WHITE);
}

static void draw_bar_fill(int samples, int total)
{
    /* remaining green portion (drains left-to-right as time passes) */
    int remaining = (total - samples) * BAR_W / total;
    int used      = BAR_W - remaining;

    if (remaining > 0)
        fill_rect(BAR_X,          BAR_Y, remaining, BAR_H, COLOR_GREEN);
    if (used > 0)
        fill_rect(BAR_X + remaining, BAR_Y, used,   BAR_H, COLOR_BLACK);
}

/* ── Public API ─────────────────────────────────────────────────────────── */

void vga_init(void)
{
    *(pixel_ctrl_ptr + 1) = (int)Buffer1;
    wait_for_vsync();
    pixel_buffer_start = *pixel_ctrl_ptr;

    *(pixel_ctrl_ptr + 1) = (int)Buffer2;
    pixel_buffer_start = *(pixel_ctrl_ptr + 1);

    clear_screen(COLOR_BLACK);
    wait_for_vsync();
    pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    clear_screen(COLOR_BLACK);

    thinking_index = 0;
}

/* ── STATE 1 entry — draw full screen, bar starts full ──────────────────── */
void vga_state_listening(void)
{
    int buf;
    for (buf = 0; buf < 2; buf++) {
        draw_listening_base();
        draw_bar_fill(0, 1);      /* samples=0, total=1 → full green bar */
        wait_for_vsync();
        pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    }
    play_tone(220, 100);
}

/* ── STATE 1 update — called every ~800 samples from main loop ──────────── */
/*
 * Redraws the full frame so both buffers stay coherent.
 * Called in main.c wherever LED[1] is pulsed.
 */
void vga_update_listening(int samples, int total)
{
    draw_listening_base();
    draw_bar_fill(samples, total);
    wait_for_vsync();
    pixel_buffer_start = *(pixel_ctrl_ptr + 1);
}

/* ── STATE 2: PROCESSING ────────────────────────────────────────────────── */
void vga_state_processing(void)
{
    static int dot_phase = 0;
    static const char *dots[3] = { ".  ", ".. ", "..." };
    const char *word = thinking_words[thinking_index % NUM_THINKING_WORDS];
    int freq = proc_freqs[thinking_index % 3];
    int buf;

    thinking_index++;

    for (buf = 0; buf < 2; buf++) {
        clear_screen(COLOR_PURPLE);
        draw_string_centered(90,  word,              COLOR_WHITE, 2);
        draw_string_centered(115, dots[dot_phase % 3], COLOR_CYAN, 2);
        wait_for_vsync();
        pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    }
    dot_phase++;
    play_tone(freq, 80);
}

/* ── STATE 3: SUCCESS ───────────────────────────────────────────────────── */
void vga_state_success(void)
{
    int buf;
    for (buf = 0; buf < 2; buf++) {
        clear_screen(COLOR_GREEN);
        draw_string_centered(90,  "PASSWORD RECOGNIZED", COLOR_BLACK, 2);
        draw_string_centered(115, "YOU MAY PROCEED",     COLOR_BLACK, 2);
        wait_for_vsync();
        pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    }
    play_tone(523, 150);
    play_tone(659, 150);
    play_tone(784, 250);
    busy_wait(140000000);
    thinking_index = 0;
}

/* ── STATE 4: FAILURE ───────────────────────────────────────────────────── */
void vga_state_failure(void)
{
    int buf;
    for (buf = 0; buf < 2; buf++) {
        clear_screen(COLOR_RED);
        draw_string_centered(90,  "WRONG PASSWORD",     COLOR_WHITE, 2);
        draw_string_centered(115, "BE GONE, INTRUDER!", COLOR_WHITE, 2);
        wait_for_vsync();
        pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    }
    play_tone(392, 200);
    play_tone(294, 300);
    busy_wait(150000000);
    thinking_index = 0;
}

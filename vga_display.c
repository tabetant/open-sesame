/*
 * vga_display.c — VGA pixel graphics + WM8731 audio for open-sesame gate.
 *
 * Follows ECE243 part3.c conventions exactly:
 *   - Buffer1/Buffer2 static arrays, pixel_buffer_start global
 *   - wait_for_vsync() polling pattern
 *   - draw_pixel() via (pixel_buffer_start + (y<<10) + (x<<1))
 *   - Bresenham draw_line()
 *   - 8 bouncing boxes joined by lines (Lab 7 Part 3 animation)
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
static volatile int   *pixel_ctrl_ptr = (volatile int   *)0xFF203020;

struct AUDIO_REG {
    volatile unsigned int control;
    volatile unsigned char rarc;
    volatile unsigned char ralc;
    volatile unsigned char wsrc;
    volatile unsigned char wslc;
    volatile int ldata;
    volatile int rdata;
};
static volatile struct AUDIO_REG *audio_hw = (volatile struct AUDIO_REG *)0xFF203040;

/* ── 5×7 bitmap font ────────────────────────────────────────────────────── */
/* Each character: 5 bytes, one per column, bits 0-6 = rows top-to-bottom   */
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
    *pixel_ctrl_ptr = 1;               /* swap front/back */
    status = *(pixel_ctrl_ptr + 3);    /* read status register */
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
    int x = x0, y = y0;

    dx = (dx < 0) ? -dx : dx;
    dy = (dy < 0) ? -dy : dy;

    if (dx >= dy) {
        int e = 2 * dy - dx;
        int i;
        for (i = 0; i <= dx; i++) {
            draw_pixel(x, y, color);
            if (e > 0) { y += sy; e -= 2 * dx; }
            e += 2 * dy;
            x += sx;
        }
    } else {
        int e = 2 * dx - dy;
        int i;
        for (i = 0; i <= dy; i++) {
            draw_pixel(x, y, color);
            if (e > 0) { x += sx; e -= 2 * dy; }
            e += 2 * dx;
            y += sy;
        }
    }
}

static void draw_rect(int x, int y, int w, int h, short int color)
{
    int i;
    for (i = x; i < x + w; i++) {
        draw_pixel(i, y,         color);
        draw_pixel(i, y + h - 1, color);
    }
    for (i = y; i < y + h; i++) {
        draw_pixel(x,         i, color);
        draw_pixel(x + w - 1, i, color);
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
/* scale: each font pixel is drawn as scale×scale screen pixels */

static void draw_char(int x, int y, char c, short int color, int scale)
{
    int col, row, sx, sy;
    int idx;

    if (c < ' ' || c > '_')  /* clip to table range */
        c = ' ';
    idx = (int)(c - ' ');

    for (col = 0; col < 5; col++) {
        unsigned char col_bits = font5x7[idx][col];
        for (row = 0; row < 7; row++) {
            if (col_bits & (1 << row)) {
                for (sy = 0; sy < scale; sy++)
                    for (sx = 0; sx < scale; sx++)
                        draw_pixel(x + col * scale + sx,
                                   y + row * scale + sy,
                                   color);
            }
        }
    }
}

static int str_len(const char *s)
{
    int n = 0;
    while (*s++) n++;
    return n;
}

/* Draw string; returns pixel width of rendered text */
static void draw_string(int x, int y, const char *s, short int color, int scale)
{
    int char_w = 6 * scale;   /* 5 pixel cols + 1 gap */
    while (*s) {
        draw_char(x, y, *s, color, scale);
        x += char_w;
        s++;
    }
}

/* Center a string horizontally on screen at row y */
static void draw_string_centered(int y, const char *s, short int color, int scale)
{
    int char_w   = 6 * scale;
    int total_w  = str_len(s) * char_w;
    int x        = (SCREEN_W - total_w) / 2;
    draw_string(x, y, s, color, scale);
}

/* ── Audio — square-wave tone synthesis ─────────────────────────────────── */
/*
 * Sample rate is 8000 Hz (codec already configured).
 * For frequency f: half-period = 8000 / (2*f) samples.
 * We write samples directly; the FIFO stalls us if full.
 */
static void play_tone(int freq_hz, int duration_ms)
{
    int total_samples = (8000 * duration_ms) / 1000;
    int half_period   = (freq_hz > 0) ? (8000 / (2 * freq_hz)) : 1;
    int sample        = 0;
    int phase         = 0;

    for (sample = 0; sample < total_samples; sample++) {
        int val = (phase < half_period) ? 0x007FFFFF : (int)0xFF800001;
        while (!audio_hw->wsrc);   /* wait for FIFO space */
        audio_hw->ldata = val;
        audio_hw->rdata = val;
        phase++;
        if (phase >= 2 * half_period)
            phase = 0;
    }
}

/* ── Bouncing-box animation state (Lab 7 Part 3 style) ──────────────────── */
#define NUM_BOXES  8
#define BOX_SIZE   10

static const short int box_colors[NUM_BOXES] = {
    COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW,
    COLOR_CYAN, COLOR_MAGENTA, COLOR_ORANGE, COLOR_WHITE
};

typedef struct {
    int x, y;
    int dx, dy;
} Box;

static Box boxes[NUM_BOXES];

static void init_boxes(void)
{
    /* Spread initial positions evenly; all move diagonally */
    int i;
    for (i = 0; i < NUM_BOXES; i++) {
        boxes[i].x  = 20 + i * 35;
        boxes[i].y  = 20 + i * 22;
        boxes[i].dx = (i % 2 == 0) ?  2 : -2;
        boxes[i].dy = (i % 3 == 0) ?  1 : -1;
    }
}

/* Erase boxes at their current positions, move them, redraw.
 * bg_color is used to erase (matches background). */
static void update_boxes(short int bg_color)
{
    int i;

    /* Erase old positions */
    for (i = 0; i < NUM_BOXES; i++)
        fill_rect(boxes[i].x, boxes[i].y, BOX_SIZE, BOX_SIZE, bg_color);

    /* Erase old lines */
    for (i = 0; i < NUM_BOXES; i++) {
        int j = (i + 1) % NUM_BOXES;
        draw_line(boxes[i].x + BOX_SIZE/2, boxes[i].y + BOX_SIZE/2,
                  boxes[j].x + BOX_SIZE/2, boxes[j].y + BOX_SIZE/2,
                  bg_color);
    }

    /* Move and bounce */
    for (i = 0; i < NUM_BOXES; i++) {
        boxes[i].x += boxes[i].dx;
        boxes[i].y += boxes[i].dy;

        if (boxes[i].x <= 0 || boxes[i].x >= SCREEN_W - BOX_SIZE)
            boxes[i].dx = -boxes[i].dx;
        if (boxes[i].y <= 0 || boxes[i].y >= SCREEN_H - BOX_SIZE)
            boxes[i].dy = -boxes[i].dy;

        /* clamp */
        if (boxes[i].x < 0) boxes[i].x = 0;
        if (boxes[i].x > SCREEN_W - BOX_SIZE) boxes[i].x = SCREEN_W - BOX_SIZE;
        if (boxes[i].y < 0) boxes[i].y = 0;
        if (boxes[i].y > SCREEN_H - BOX_SIZE) boxes[i].y = SCREEN_H - BOX_SIZE;
    }

    /* Draw new lines */
    for (i = 0; i < NUM_BOXES; i++) {
        int j = (i + 1) % NUM_BOXES;
        draw_line(boxes[i].x + BOX_SIZE/2, boxes[i].y + BOX_SIZE/2,
                  boxes[j].x + BOX_SIZE/2, boxes[j].y + BOX_SIZE/2,
                  box_colors[i]);
    }

    /* Draw new box outlines */
    for (i = 0; i < NUM_BOXES; i++)
        draw_rect(boxes[i].x, boxes[i].y, BOX_SIZE, BOX_SIZE, box_colors[i]);
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

/* Processing beep frequencies: A, C, E cycling */
static const int proc_freqs[3] = { 440, 523, 659 };

/* ── Simple busy-wait delay (approximate) ──────────────────────────────── */
/*
 * The board runs at ~100 MHz. We need short pauses inside vga_state_*
 * without blocking the audio capture loop for long.  The hold delays
 * (success/failure 2 s) are intentionally blocking — main.c replaces
 * the old delay(3000000) call with these functions.
 */
static void busy_wait(int count)
{
    volatile int c = count;
    while (c--);
}

/* ── Public API ─────────────────────────────────────────────────────────── */

void vga_init(void)
{
    /* Point back-buffer at Buffer1, front-buffer at Buffer2 */
    *(pixel_ctrl_ptr + 1) = (int)Buffer1;   /* back buffer address  */
    wait_for_vsync();                        /* swap: front=Buffer1  */
    pixel_buffer_start = *pixel_ctrl_ptr;   /* now drawing into front (Buffer1) */

    /* Set back buffer to Buffer2 */
    *(pixel_ctrl_ptr + 1) = (int)Buffer2;
    pixel_buffer_start = *(pixel_ctrl_ptr + 1);  /* draw into back (Buffer2) */

    clear_screen(COLOR_BLACK);
    wait_for_vsync();
    pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    clear_screen(COLOR_BLACK);

    init_boxes();
    thinking_index = 0;
}

/* ── STATE 1: IDLE / LISTENING ──────────────────────────────────────────── */
/*
 * Called once on entry.  Draws the static text on both buffers so it
 * persists, then runs one frame of box animation and plays one cycle of
 * the ambient hum.  main.c does NOT call this every frame — it calls it
 * once when returning to idle, so we do one full rendered frame here.
 *
 * The box animation advances every time vga_state_listening() is called
 * (which happens once per ~2-second inference stride while idle).
 */
void vga_state_listening(void)
{
    int buf;

    /* Draw into both buffers so text is visible on both frames */
    for (buf = 0; buf < 2; buf++) {
        clear_screen(COLOR_BLACK);

        /* Title line — scale 2 */
        draw_string_centered(70,  "YOU HAVE REACHED THE GATE", COLOR_WHITE,  2);
        /* Subtitle line — scale 1 */
        draw_string_centered(105, "SPEAK THE PASSWORD",        COLOR_YELLOW, 2);

        update_boxes(COLOR_BLACK);

        wait_for_vsync();
        pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    }

    /* Ambient hum: short 220 Hz burst (non-blocking feel — 100 ms) */
    play_tone(220, 100);
}

/* ── STATE 2: PROCESSING ────────────────────────────────────────────────── */
/*
 * Called once per inference cycle (every ~2 s of audio).
 * Cycles through thinking words, advances dot animation.
 */
void vga_state_processing(void)
{
    static int dot_phase = 0;
    static const char *dots[3] = { ".  ", ".. ", "..." };
    const char *word;
    int freq;
    int buf;

    word = thinking_words[thinking_index % NUM_THINKING_WORDS];
    thinking_index++;

    freq = proc_freqs[(thinking_index) % 3];

    for (buf = 0; buf < 2; buf++) {
        /* Deep blue/purple background */
        clear_screen(COLOR_PURPLE);

        /* Main thinking word — scale 2 */
        draw_string_centered(90,  word,          COLOR_WHITE,  2);
        /* Dot animation — scale 2 */
        draw_string_centered(115, dots[dot_phase % 3], COLOR_CYAN, 2);

        wait_for_vsync();
        pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    }

    dot_phase++;

    /* Ascending beep on each new word */
    play_tone(freq, 80);
}

/* ── STATE 3: SUCCESS ───────────────────────────────────────────────────── */
void vga_state_success(void)
{
    int buf, i;

    for (buf = 0; buf < 2; buf++) {
        clear_screen(COLOR_GREEN);
        draw_string_centered(90,  "PASSWORD RECOGNIZED", COLOR_BLACK, 2);
        draw_string_centered(115, "YOU MAY PROCEED",     COLOR_BLACK, 2);

        wait_for_vsync();
        pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    }

    /* Triumphant C-E-G chime */
    play_tone(523, 150);
    play_tone(659, 150);
    play_tone(784, 250);

    /* Hold for ~2 seconds (remaining time after tones ~0.55 s already played) */
    busy_wait(140000000);   /* ~1.4 s at 100 MHz */

    thinking_index = 0;
}

/* ── STATE 4: FAILURE ───────────────────────────────────────────────────── */
void vga_state_failure(void)
{
    int buf;

    for (buf = 0; buf < 2; buf++) {
        clear_screen(COLOR_RED);
        draw_string_centered(90,  "WRONG PASSWORD",    COLOR_WHITE, 2);
        draw_string_centered(115, "BE GONE, INTRUDER!", COLOR_WHITE, 2);

        wait_for_vsync();
        pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    }

    /* Descending G → D rejection */
    play_tone(392, 200);
    play_tone(294, 300);

    /* Hold for ~2 seconds (tones = 0.5 s, so wait another 1.5 s) */
    busy_wait(150000000);   /* ~1.5 s at 100 MHz */

    thinking_index = 0;
}

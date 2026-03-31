/*
 * cpulator.c — Standalone VGA + audio test for CPUlator (DE1-SoC simulator)
 *
 * USAGE: Paste the entire contents of this file into CPUlator as a single
 * compilation unit. It has its own main(). Do NOT add it to the Makefile.
 *
 * What it does:
 *   - Boots into STATE 1 (idle/bouncing boxes) for ~3 seconds
 *   - Cycles through STATE 2 (processing words) 5 times
 *   - Shows STATE 3 (success / green) for 2 seconds
 *   - Returns to STATE 1 briefly
 *   - Cycles STATE 2 again 3 times
 *   - Shows STATE 4 (failure / red) for 2 seconds
 *   - Loops forever from STATE 1
 *
 * All four visual states and all audio tones are exercised in one run.
 *
 * CPUlator notes:
 *   - VGA pixel buffer controller: 0xFF203020  (same as real board)
 *   - Audio FIFO write: 0xFF203040  (wsrc = write-space available)
 *   - Audio FIFO wait has an iteration cap so it never hangs if audio
 *     is disabled in the simulator tab.
 *   - printf goes to the CPUlator terminal via JTAG UART emulation.
 */

#include <stdio.h>

/* ── Hardware base addresses ─────────────────────────────────────────────── */
#define PIXEL_BUF_CTRL_BASE  0xFF203020
#define AUDIO_BASE           0xFF203040
#define LED_BASE             0xFF200000

/* ── Screen dimensions ───────────────────────────────────────────────────── */
#define SCREEN_W  320
#define SCREEN_H  240

/* ── RGB565 color constants ──────────────────────────────────────────────── */
#define COLOR_BLACK   0x0000
#define COLOR_WHITE   0xFFFF
#define COLOR_RED     0xF800
#define COLOR_GREEN   0x07E0
#define COLOR_BLUE    0x001F
#define COLOR_YELLOW  0xFFE0
#define COLOR_CYAN    0x07FF
#define COLOR_MAGENTA 0xF81F
#define COLOR_PURPLE  0x4010
#define COLOR_ORANGE  0xFC00

/* ── Double-buffer globals (ECE243 part3.c style) ────────────────────────── */
int pixel_buffer_start;

static short int Buffer1[240][512];
static short int Buffer2[240][512];

/* ── Audio register layout ───────────────────────────────────────────────── */
struct AUDIO_REG {
    volatile unsigned int  control;
    volatile unsigned char rarc;
    volatile unsigned char ralc;
    volatile unsigned char wsrc;   /* write space available */
    volatile unsigned char wslc;
    volatile int ldata;
    volatile int rdata;
};
static volatile struct AUDIO_REG *audio_hw =
    (volatile struct AUDIO_REG *)AUDIO_BASE;

/* ── Hardware register pointer ───────────────────────────────────────────── */
static volatile int *pixel_ctrl_ptr = (volatile int *)PIXEL_BUF_CTRL_BASE;

/* ── 5×7 bitmap font (ASCII 32 ' ' through 95 '_') ──────────────────────── */
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

/* ══════════════════════════════════════════════════════════════════════════
 * Low-level draw helpers  (ECE243 part3.c style)
 * ══════════════════════════════════════════════════════════════════════════ */

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

static void draw_line(int x0, int y0, int x1, int y1, short int color)
{
    int dx = x1 - x0;
    int dy = y1 - y0;
    int sx = (dx > 0) ? 1 : -1;
    int sy = (dy > 0) ? 1 : -1;
    int x = x0, y = y0;
    int i;

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

/* ══════════════════════════════════════════════════════════════════════════
 * Text rendering
 * ══════════════════════════════════════════════════════════════════════════ */

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
    while (*s) { draw_char(x, y, *s, color, scale); x += 6*scale; s++; }
}

static void draw_string_centered(int y, const char *s, short int color, int scale)
{
    int x = (SCREEN_W - str_len(s) * 6 * scale) / 2;
    draw_string(x, y, s, color, scale);
}

/* ══════════════════════════════════════════════════════════════════════════
 * Audio — square-wave tones
 * Capped at MAX_AUDIO_ITER iterations so it never hangs if the CPUlator
 * audio tab is disabled.
 * ══════════════════════════════════════════════════════════════════════════ */
#define MAX_AUDIO_ITER 200000

static void play_tone(int freq_hz, int duration_ms)
{
    int total   = (8000 * duration_ms) / 1000;
    int half    = (freq_hz > 0) ? (8000 / (2 * freq_hz)) : 1;
    int sample, phase = 0, guard;

    for (sample = 0; sample < total; sample++) {
        int val = (phase < half) ? 0x007FFFFF : (int)0xFF800001;
        guard = 0;
        while (!audio_hw->wsrc && guard < MAX_AUDIO_ITER) guard++;
        if (guard < MAX_AUDIO_ITER) {
            audio_hw->ldata = val;
            audio_hw->rdata = val;
        }
        phase++;
        if (phase >= 2 * half) phase = 0;
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * Bouncing boxes (Lab 7 Part 3 animation)
 * ══════════════════════════════════════════════════════════════════════════ */
#define NUM_BOXES  8
#define BOX_SIZE   10

static const short int box_colors[NUM_BOXES] = {
    COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW,
    COLOR_CYAN, COLOR_MAGENTA, COLOR_ORANGE, COLOR_WHITE
};

typedef struct { int x, y, dx, dy; } Box;
static Box boxes[NUM_BOXES];

static void init_boxes(void)
{
    int i;
    for (i = 0; i < NUM_BOXES; i++) {
        boxes[i].x  = 20 + i * 35;
        boxes[i].y  = 20 + i * 22;
        boxes[i].dx = (i % 2 == 0) ?  2 : -2;
        boxes[i].dy = (i % 3 == 0) ?  1 : -1;
    }
}

static void update_boxes(short int bg)
{
    int i;
    /* erase old */
    for (i = 0; i < NUM_BOXES; i++)
        fill_rect(boxes[i].x, boxes[i].y, BOX_SIZE, BOX_SIZE, bg);
    for (i = 0; i < NUM_BOXES; i++) {
        int j = (i+1) % NUM_BOXES;
        draw_line(boxes[i].x+BOX_SIZE/2, boxes[i].y+BOX_SIZE/2,
                  boxes[j].x+BOX_SIZE/2, boxes[j].y+BOX_SIZE/2, bg);
    }
    /* move + bounce */
    for (i = 0; i < NUM_BOXES; i++) {
        boxes[i].x += boxes[i].dx;
        boxes[i].y += boxes[i].dy;
        if (boxes[i].x <= 0 || boxes[i].x >= SCREEN_W - BOX_SIZE) boxes[i].dx = -boxes[i].dx;
        if (boxes[i].y <= 0 || boxes[i].y >= SCREEN_H - BOX_SIZE) boxes[i].dy = -boxes[i].dy;
        if (boxes[i].x < 0) boxes[i].x = 0;
        if (boxes[i].x > SCREEN_W-BOX_SIZE) boxes[i].x = SCREEN_W-BOX_SIZE;
        if (boxes[i].y < 0) boxes[i].y = 0;
        if (boxes[i].y > SCREEN_H-BOX_SIZE) boxes[i].y = SCREEN_H-BOX_SIZE;
    }
    /* draw new */
    for (i = 0; i < NUM_BOXES; i++) {
        int j = (i+1) % NUM_BOXES;
        draw_line(boxes[i].x+BOX_SIZE/2, boxes[i].y+BOX_SIZE/2,
                  boxes[j].x+BOX_SIZE/2, boxes[j].y+BOX_SIZE/2, box_colors[i]);
    }
    for (i = 0; i < NUM_BOXES; i++)
        draw_rect(boxes[i].x, boxes[i].y, BOX_SIZE, BOX_SIZE, box_colors[i]);
}

/* ══════════════════════════════════════════════════════════════════════════
 * Thinking words
 * ══════════════════════════════════════════════════════════════════════════ */
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
static const int proc_freqs[3] = { 440, 523, 659 };

/* ══════════════════════════════════════════════════════════════════════════
 * Busy-wait delay (~100 MHz assumed)
 * ══════════════════════════════════════════════════════════════════════════ */
static void busy_wait(int count)
{
    volatile int c = count; while (c--);
}

/* ══════════════════════════════════════════════════════════════════════════
 * VGA init  (ECE243 part3.c double-buffer init)
 * ══════════════════════════════════════════════════════════════════════════ */
static void vga_init(void)
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

    init_boxes();
}

/* ══════════════════════════════════════════════════════════════════════════
 * State renderers
 * ══════════════════════════════════════════════════════════════════════════ */

static void state_listening(void)
{
    int buf;
    printf("[STATE 1] IDLE / LISTENING\n");
    for (buf = 0; buf < 2; buf++) {
        clear_screen(COLOR_BLACK);
        draw_string_centered(70,  "YOU HAVE REACHED THE GATE", COLOR_WHITE,  2);
        draw_string_centered(105, "SPEAK THE PASSWORD",        COLOR_YELLOW, 2);
        update_boxes(COLOR_BLACK);
        wait_for_vsync();
        pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    }
    play_tone(220, 100);
}

static void state_processing(int cycle)
{
    static const char *dots[3] = { ".  ", ".. ", "..." };
    const char *word = thinking_words[cycle % NUM_THINKING_WORDS];
    int freq = proc_freqs[cycle % 3];
    int buf;

    printf("[STATE 2] PROCESSING — %s\n", word);
    for (buf = 0; buf < 2; buf++) {
        clear_screen(COLOR_PURPLE);
        draw_string_centered(90,  word,              COLOR_WHITE, 2);
        draw_string_centered(115, dots[cycle % 3],   COLOR_CYAN,  2);
        wait_for_vsync();
        pixel_buffer_start = *(pixel_ctrl_ptr + 1);
    }
    play_tone(freq, 80);
}

static void state_success(void)
{
    int buf;
    printf("[STATE 3] SUCCESS\n");
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
}

static void state_failure(void)
{
    int buf;
    printf("[STATE 4] FAILURE\n");
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
}

/* ══════════════════════════════════════════════════════════════════════════
 * main — demo loop
 *
 * Sequence:
 *   STATE1 (3 frames of boxes)
 *   STATE2 x5  (simulates ~10 s of CNN)
 *   STATE3      (success)
 *   STATE1 (2 frames)
 *   STATE2 x3
 *   STATE4      (failure)
 *   ... repeat forever
 * ══════════════════════════════════════════════════════════════════════════ */
int main(void)
{
    volatile unsigned int *leds = (volatile unsigned int *)LED_BASE;
    int cycle = 0;
    int round = 0;

    printf("[BOOT] CPUlator VGA test starting\n");

    vga_init();
    *leds = 0x001;

    while (1) {
        int i;

        /* ── STATE 1: idle — run a few animation frames ── */
        printf("[ROUND %d] Entering STATE 1\n", round);
        for (i = 0; i < 3; i++) {
            state_listening();
            busy_wait(5000000);   /* short pause between frames */
        }

        /* ── STATE 2: processing cycles — odd rounds do 5, even do 3 ── */
        {
            int n = (round % 2 == 0) ? 5 : 3;
            printf("[ROUND %d] Running %d processing cycles\n", round, n);
            *leds = 0x005;
            for (i = 0; i < n; i++) {
                state_processing(cycle++);
                busy_wait(8000000);
            }
            *leds = 0x011;
        }

        /* ── STATE 3 or 4 alternating by round ── */
        if (round % 2 == 0) {
            state_success();
            *leds = 0x3FF;   /* all LEDs on = success */
        } else {
            state_failure();
            *leds = 0x001;   /* back to boot LED */
        }

        round++;
        busy_wait(3000000);
    }

    return 0;
}

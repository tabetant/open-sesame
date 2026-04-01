#ifndef VGA_DISPLAY_H
#define VGA_DISPLAY_H

/* ── VGA pixel buffer ───────────────────────────────────────────────────────
 * 320x240, 16-bit RGB565, double-buffered.
 * Row stride = 1024 bytes (y << 10), pixel stride = 2 bytes (x << 1).
 * Matches ECE243 part3.c conventions exactly.
 */

/* 16-bit RGB565 color constants */
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

/* Screen dimensions */
#define SCREEN_W  320
#define SCREEN_H  240

/* State machine — call these from main.c */
void vga_init(void);              /* call once at boot after setup_gpio()  */
void vga_state_listening(void);   /* idle / waiting for voice             */
void vga_update_listening(int samples, int total); /* call every ~800 samples to update countdown bar */
void vga_state_processing(int phase); /* 0=before MFCC, 1=after MFCC, 2=pre-CNN wait, 3=before CNN */
void vga_pause(int duration_ms);      /* accurate silent wait using audio FIFO clock              */
void vga_state_success(void);     /* prob > 0.27 — holds ~2 s             */
void vga_state_failure(void);     /* prob <= 0.27 — holds ~2 s            */

#endif /* VGA_DISPLAY_H */

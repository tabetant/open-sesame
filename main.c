/*
 * main.c — Voice-Activated LEGO Gate
 * ECE243 Project, University of Toronto
 *
 * Pipeline:
 *   WM8731 audio codec (8 kHz) → circular buffer → MFCC extraction
 *   → CNN inference → if P("open sesame") > 0.9 → open LEGO gate
 *
 * All heavy data (model weights, MFCC tables) lives in SDRAM.
 * On-chip SRAM is used only for stack and small locals.
 */

#include "address_map.h"
#include "lego_motor.h"
#include "mfcc.h"
#include "inference.h"

/* ── Audio hardware ─────────────────────────────────────────────────────── */
struct AUDIO_T {
    volatile unsigned int control;  /* offset 0x00 */
    volatile unsigned char rarc;    /* read samples available (right) */
    volatile unsigned char ralc;    /* read samples available (left)  */
    volatile unsigned char wsrc;    /* write space available (right)  */
    volatile unsigned char wslc;    /* write space available (left)   */
    volatile int ldata;             /* left  data register */
    volatile int rdata;             /* right data register */
};

/* ── I2C / Audio codec config ───────────────────────────────────────────── */
struct I2C_T {
    volatile unsigned int prescale_lo;  /* 0x00 — clock prescaler low  */
    volatile unsigned int prescale_hi;  /* 0x04 — clock prescaler high */
    volatile unsigned int control;      /* 0x08 — enable / IRQ enable  */
    volatile unsigned int data;         /* 0x0C — TX/RX data           */
    volatile unsigned int cmd_status;   /* 0x10 — command / status     */
};

/* I2C command/status bits */
#define I2C_START    0x90   /* STA + WR */
#define I2C_WRITE    0x10   /* WR       */
#define I2C_STOP     0x40   /* STO      */
#define I2C_RXACK    0x80   /* RX ACK flag in status */
#define I2C_TIP      0x02   /* Transfer in progress  */
#define I2C_BUSY     0x40   /* Bus busy              */

/* WM8731 I2C address (CSB=0) */
#define WM8731_ADDR  0x1A

static struct I2C_T  *i2c   = (struct I2C_T  *)AV_CONFIG_BASE;
static struct AUDIO_T *audio = (struct AUDIO_T *)AUDIO_BASE;

/* Wait until the I2C transfer-in-progress bit clears */
static void i2c_wait(void)
{
    while (i2c->cmd_status & I2C_TIP);
}

/* Send one byte over I2C with the given command flags */
static void i2c_write_byte(unsigned char data, unsigned char cmd)
{
    i2c->data    = data;
    i2c->cmd_status = cmd;
    i2c_wait();
}

/* Write one WM8731 register.
 * The WM8731 protocol: one 2-byte transfer over I2C.
 * Byte 0: [6:1]=reg_addr, [0]=data[8]
 * Byte 1: data[7:0]  */
static void wm8731_write(unsigned char reg, unsigned short val)
{
    unsigned char b0 = (reg << 1) | ((val >> 8) & 0x01);
    unsigned char b1 = val & 0xFF;

    i2c_write_byte((WM8731_ADDR << 1), I2C_START);  /* address + WRITE */
    i2c_write_byte(b0, I2C_WRITE);
    i2c_write_byte(b1, I2C_WRITE | I2C_STOP);
}

/*
 * init_audio_codec()
 *
 * Configure the WM8731 for:
 *   - 8 kHz sample rate
 *   - 16-bit, I2S format
 *   - Line input unmuted
 *   - ADC (microphone) and DAC powered on
 *
 * Prescaler: 50 MHz system clock / (5 * 400 kHz I2C) - 1 = 24
 */
static void init_audio_codec(void)
{
    /* Set I2C prescaler for 400 kHz at 50 MHz system clock */
    i2c->prescale_lo = 24;
    i2c->prescale_hi = 0;
    i2c->control     = 0x80;   /* Enable I2C core */

    /* Reset the WM8731 */
    wm8731_write(15, 0x000);

    /* R0: Left  line input — 0 dB, unmute */
    wm8731_write(0, 0x017);
    /* R1: Right line input — 0 dB, unmute */
    wm8731_write(1, 0x017);
    /* R4: Analogue audio path — MIC input selected, ADC on */
    wm8731_write(4, 0x014);
    /* R5: Digital audio path — no HPF */
    wm8731_write(5, 0x000);
    /* R6: Power down — everything on */
    wm8731_write(6, 0x000);
    /* R7: Digital audio interface — I2S, 16-bit */
    wm8731_write(7, 0x00A);
    /* R8: Sampling control — 8 kHz (USB mode, SR=0011, BOSR=0)
     *     CLKODIV2=0, CLKIDIV2=0, SR=0011, BOSR=0, USB/Normal=1 */
    wm8731_write(8, 0x00D);
    /* R9: Active — start the codec */
    wm8731_write(9, 0x001);
}

/* ── Circular buffer (2 seconds = 16000 samples at 8 kHz) ──────────────── */
#define CIRC_BUF_SIZE  16000
#define INFERENCE_STRIDE 8000   /* run inference every 1 second of new audio */

/* Audio window fed to MFCC: N_FFT + HOP_LENGTH*(N_FRAMES-1) = 15880 samples */
#define AUDIO_WINDOW_LEN  15880

static float circ_buf[CIRC_BUF_SIZE];
static int   circ_write  = 0;
static int   new_samples = 0;

/* MFCC feature buffer and flat audio window — static to stay off the stack */
static float mfcc_buf[N_MFCC][N_FRAMES];
static float audio_window[AUDIO_WINDOW_LEN];
static float prob[2];

/* ── Gate control ───────────────────────────────────────────────────────── */
/* Tune these values on real hardware: 1 count ≈ 1 cycle at ~100 MHz.
 * Start with OPEN_DELAY=50000000 (~0.5 s) and increase until gate fully opens. */
#define OPEN_DELAY  50000000
#define HOLD_DELAY  20000000

static void open_and_close_gate(void)
{
    spin_motor(0, 1);       /* open  — clockwise     */
    delay(OPEN_DELAY);
    stop_motor(0);
    delay(HOLD_DELAY);
    spin_motor(0, 0);       /* close — counter-clockwise */
    delay(OPEN_DELAY);
    stop_motor(0);
}

/* ── Main ───────────────────────────────────────────────────────────────── */
int main(void)
{
    int i;

    setup_gpio();
    stop_all_motors();
    init_audio_codec();

    /* Signal ready via LEDs */
    volatile unsigned int *leds = (volatile unsigned int *)LED_BASE;
    *leds = 0x1;   /* LED[0] on = codec initialised, waiting for command */

    while (1) {
        /* Poll audio FIFO — rarc is non-zero when at least one sample is ready */
        if (audio->rarc) {
            /* WM8731 via DE1-SoC audio core: 16-bit sample is left-justified
             * in a 32-bit register (bits 31:16). Shift to get signed 16-bit,
             * then normalise to [-1.0, 1.0]. */
            int raw_left = audio->ldata;
            (void)audio->rdata;   /* discard right channel */

            float sample = (float)(raw_left >> 16) / 32768.0f;

            circ_buf[circ_write] = sample;
            circ_write = (circ_write + 1) % CIRC_BUF_SIZE;
            new_samples++;

            if (new_samples >= INFERENCE_STRIDE) {
                new_samples = 0;

                /* Copy the most recent AUDIO_WINDOW_LEN samples from the
                 * circular buffer into a flat array for compute_mfcc().     */
                int start = (circ_write - AUDIO_WINDOW_LEN + CIRC_BUF_SIZE)
                            % CIRC_BUF_SIZE;
                for (i = 0; i < AUDIO_WINDOW_LEN; i++)
                    audio_window[i] = circ_buf[(start + i) % CIRC_BUF_SIZE];

                /* MFCC extraction + normalization */
                compute_mfcc(audio_window, mfcc_buf);

                /* CNN inference */
                run_inference((const float (*)[N_FRAMES])mfcc_buf, prob);

                /* Trigger gate if confidence > 0.9 */
                if (prob[1] > 0.9f) {
                    *leds = 0x3FF;          /* all LEDs on = gate triggered */
                    open_and_close_gate();
                    *leds = 0x1;            /* back to idle state           */

                    /* Flush accumulated samples during gate operation so
                     * the gate doesn't immediately re-trigger.             */
                    new_samples = 0;
                }
            }
        }
    }

    return 0;
}

#include "inference.h"
#include <math.h>

/* ── Layer output dimensions ───────────────────────────────────────────────
 * Input:             [13][99][1]
 * After Conv1+ReLU:  [13][99][32]   = 41184 floats
 * After MaxPool1:    [ 6][49][32]   = 9408  floats
 * After Conv2+ReLU:  [ 6][49][64]   = 18816 floats
 * After MaxPool2:    [ 3][24][64]   = 4608  floats  ← Flatten input
 * After Dense1+ReLU: [64]
 * After Dense2:      [2]  → Softmax → prob[2]
 * ───────────────────────────────────────────────────────────────────────── */

#define CONV1_H   13
#define CONV1_W   99
#define CONV1_C   32
#define POOL1_H   6
#define POOL1_W   49
#define CONV2_C   64
#define POOL2_H   3
#define POOL2_W   24
#define FLAT_SIZE 4608   /* 3 * 24 * 64 */
#define DENSE1_OUT 64
#define DENSE2_OUT 2

/* Ping-pong activation buffers — static so they live in SDRAM, not the stack.
 * Largest activation: Conv1 output 13*99*32 = 41184 floats (160 KB each).   */
static float act_a[41184];
static float act_b[41184];

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

/* conv2d_same: padding='same' with stride=1, kernel 3x3
 * in  layout: [H][W][C_in]  (row-major)
 * out layout: [H][W][C_out]
 * w   layout: [C_out][C_in][3][3]  */
static void conv2d_same(
    const float *in,  float *out,
    const float *w,   const float *b,
    int H, int W, int C_in, int C_out,
    int apply_relu)
{
    int oh, ow, oc, ic, kh, kw;
    for (oh = 0; oh < H; oh++) {
        for (ow = 0; ow < W; ow++) {
            for (oc = 0; oc < C_out; oc++) {
                float acc = b[oc];
                for (ic = 0; ic < C_in; ic++) {
                    for (kh = 0; kh < 3; kh++) {
                        int ih = oh + kh - 1;  /* padding=1 for same */
                        if (ih < 0 || ih >= H) continue;
                        for (kw = 0; kw < 3; kw++) {
                            int iw = ow + kw - 1;
                            if (iw < 0 || iw >= W) continue;
                            float x = in[(ih * W + iw) * C_in + ic];
                            float wt = w[((oc * C_in + ic) * 3 + kh) * 3 + kw];
                            acc += x * wt;
                        }
                    }
                }
                float val = apply_relu ? relu(acc) : acc;
                out[(oh * W + ow) * C_out + oc] = val;
            }
        }
    }
}

/* maxpool2d: 2x2 pool, stride=2, no padding
 * in/out layout: [H][W][C]  */
static void maxpool2d(
    const float *in, float *out,
    int H, int W, int C)
{
    int oh, ow, oc;
    int OH = H / 2, OW = W / 2;
    for (oh = 0; oh < OH; oh++) {
        for (ow = 0; ow < OW; ow++) {
            for (oc = 0; oc < C; oc++) {
                float m = in[((oh*2)   * W + (ow*2)  ) * C + oc];
                float v;
                v = in[((oh*2)   * W + (ow*2+1)) * C + oc]; if (v > m) m = v;
                v = in[((oh*2+1) * W + (ow*2)  ) * C + oc]; if (v > m) m = v;
                v = in[((oh*2+1) * W + (ow*2+1)) * C + oc]; if (v > m) m = v;
                out[(oh * OW + ow) * C + oc] = m;
            }
        }
    }
}

/* dense: out[j] = sum_i in[i]*w[i][j] + b[j] */
static void dense(
    const float *in, float *out,
    const float *w,  const float *b,
    int in_dim, int out_dim,
    int apply_relu)
{
    int i, j;
    for (j = 0; j < out_dim; j++) {
        float acc = b[j];
        for (i = 0; i < in_dim; i++)
            acc += in[i] * w[i * out_dim + j];
        out[j] = apply_relu ? relu(acc) : acc;
    }
}

static void softmax(float *x, int n)
{
    int i;
    float max_val = x[0];
    for (i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (i = 0; i < n; i++) x[i] /= sum;
}

/* ── run_inference ───────────────────────────────────────────────────────── */
void run_inference(const float mfcc_in[N_MFCC][N_FRAMES], float prob_out[2])
{
    int h, w;

    /* Reshape mfcc_in [13][99] into act_a [13][99][1] (trivial, same data) */
    for (h = 0; h < CONV1_H; h++)
        for (w = 0; w < CONV1_W; w++)
            act_a[(h * CONV1_W + w) * 1] = mfcc_in[h][w];

    /* Conv1: [13][99][1] → [13][99][32] + ReLU */
    conv2d_same(act_a, act_b,
                &conv1_w[0][0][0][0], conv1_b,
                CONV1_H, CONV1_W, 1, CONV1_C, 1);

    /* MaxPool1: [13][99][32] → [6][49][32] */
    maxpool2d(act_b, act_a, CONV1_H, CONV1_W, CONV1_C);

    /* Conv2: [6][49][32] → [6][49][64] + ReLU */
    conv2d_same(act_a, act_b,
                &conv2_w[0][0][0][0], conv2_b,
                POOL1_H, POOL1_W, CONV1_C, CONV2_C, 1);

    /* MaxPool2: [6][49][64] → [3][24][64] = 4608 floats (Flatten) */
    maxpool2d(act_b, act_a, POOL1_H, POOL1_W, CONV2_C);

    /* Dense1: 4608 → 64 + ReLU  (Dropout skipped at inference) */
    dense(act_a, act_b,
          &dense1_w[0][0], dense1_b,
          FLAT_SIZE, DENSE1_OUT, 1);

    /* Dense2: 64 → 2 */
    dense(act_b, prob_out,
          &dense2_w[0][0], dense2_b,
          DENSE1_OUT, DENSE2_OUT, 0);

    softmax(prob_out, DENSE2_OUT);
}

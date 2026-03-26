/*
 * inference.c — CNN Forward Pass
 *
 * ── What is a CNN? ────────────────────────────────────────────────────────────
 *
 * A CNN (Convolutional Neural Network) is a type of neural network designed to
 * find patterns in 2D data.  Originally developed for images, CNNs work equally
 * well on spectrograms — which are essentially images where one axis is time and
 * the other is frequency.
 *
 * Our CNN takes a 13×99 MFCC matrix as input and outputs two probabilities:
 *   prob[0] = probability that the audio is NOT "open sesame"
 *   prob[1] = probability that the audio IS  "open sesame"
 * These always sum to 1.0 (enforced by the softmax at the end).
 *
 * The network was trained in Python using TensorFlow/Keras.  The weights (the
 * learned numbers that make it work) were exported by export_weights.py into
 * model_data.c.  This file implements the same mathematical operations that
 * Keras performs, so that the trained model can run on the bare-metal Nios V.
 *
 * ── Network architecture ──────────────────────────────────────────────────────
 *
 *   Input:              [13][99][1]   — MFCC matrix, 1 channel (greyscale)
 *   Conv2D(32, 3×3) + ReLU           — 32 learned feature detectors
 *   MaxPool2D(2×2):     [6][49][32]  — downsample by 2 in each dimension
 *   Conv2D(64, 3×3) + ReLU           — 64 learned feature detectors
 *   MaxPool2D(2×2):     [3][24][64]  — downsample again
 *   Flatten:            [4608]       — 3 × 24 × 64 = 4608 numbers
 *   Dense(64) + ReLU                 — fully-connected layer, 64 neurons
 *   (Dropout 0.5 — only during training, skipped here)
 *   Dense(2):           [2]          — 2 output scores
 *   Softmax:            [2]          — convert scores to probabilities
 *
 * ── Memory: ping-pong activation buffers ──────────────────────────────────────
 *
 * At each layer, one buffer holds the INPUT activations and the other holds
 * the OUTPUT activations.  After each layer, the roles swap.
 *
 * This is called "ping-pong buffering".  It avoids allocating a separate buffer
 * for every layer, keeping memory use to just two large buffers.
 *
 * The largest layer output is Conv1: 13 × 99 × 32 = 41,184 floats = 160 KB.
 * We size both buffers for this worst case.
 *
 * Both are declared static so they live in SDRAM (not on the stack).
 */

#include "inference.h"
#include <math.h>

/* Layer dimension constants — must match the Python model exactly.
 * If you change the model architecture and retrain, update these. */
#define CONV1_H    13   /* input height = N_MFCC (13 MFCC coefficients)     */
#define CONV1_W    99   /* input width  = N_FRAMES (99 time frames)          */
#define CONV1_C    32   /* Conv1 output channels (32 learned filters)        */
#define POOL1_H     6   /* after MaxPool1: height halved  (13/2 = 6, floor)  */
#define POOL1_W    49   /* after MaxPool1: width  halved  (99/2 = 49, floor) */
#define CONV2_C    64   /* Conv2 output channels (64 learned filters)        */
#define POOL2_H     3   /* after MaxPool2: height halved  (6/2  = 3)         */
#define POOL2_W    24   /* after MaxPool2: width  halved  (49/2 = 24, floor) */
#define FLAT_SIZE  4608 /* flatten: 3 × 24 × 64 = 4608                       */
#define DENSE1_OUT  64  /* Dense1 output size                                 */
#define DENSE2_OUT   2  /* Dense2 output size = number of classes             */

/*
 * Ping-pong activation buffers.
 *
 * act_a and act_b take turns being the input and output of each layer:
 *   Conv1:   act_a → act_b   (input: 13*99*1 = 1287,  output: 13*99*32 = 41184)
 *   Pool1:   act_b → act_a   (input: 41184,            output: 6*49*32  = 9408)
 *   Conv2:   act_a → act_b   (input: 9408,             output: 6*49*64  = 18816)
 *   Pool2:   act_b → act_a   (input: 18816,            output: 3*24*64  = 4608)
 *   Dense1:  act_a → act_b   (input: 4608,             output: 64)
 *   Dense2:  act_b → prob    (input: 64,               output: 2)
 *
 * Size both to 41184 floats (the largest single layer output).
 * 41184 × 4 bytes = 160 KB each.  Both are static → live in SDRAM.
 */
static float act_a[41184];
static float act_b[41184];


/* ═══════════════════════════════════════════════════════════════════════════
 * LAYER IMPLEMENTATIONS
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * relu — Rectified Linear Unit activation function.
 *
 * The simplest non-linearity in neural networks.  Without it, stacking many
 * layers would just be equivalent to one big linear transformation (useless).
 * ReLU introduces non-linearity by zeroing out negative values.
 *
 *   relu(x) = x  if x > 0
 *           = 0  if x ≤ 0
 *
 * Why "inline"?
 *   This function is called millions of times.  The inline keyword asks the
 *   compiler to paste the function body directly at each call site instead of
 *   performing a function call, eliminating the call overhead.
 */
static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }


/*
 * conv2d_same — 2D convolution with 'same' padding and stride 1.
 *
 * What does a convolutional layer do?
 *   It slides a small "filter" (or "kernel") — here 3×3 pixels — across the
 *   input and at each position computes a weighted sum of the input values
 *   under the filter.  The weights are the learned parameters.  Different
 *   filters learn to detect different patterns (e.g. edges, harmonics, onsets).
 *
 *   With C_out = 32 filters, the layer runs 32 different filters over the
 *   input, producing 32 output "channels" — each highlighting a different
 *   feature of the spectrogram.
 *
 * Padding = 'same':
 *   The output has the same height and width as the input.  We achieve this
 *   by padding with 1 zero on each side (since kernel = 3, padding = 1).
 *   Without padding, each convolution would shrink the spatial dimensions.
 *
 * Data layout (row-major, innermost dimension changes fastest):
 *   in / out: [H][W][C]  → element (h, w, c) is at index (h*W + w)*C + c
 *   weights:  [C_out][C_in][3][3] → precomputed layout for cache efficiency
 *
 * Bias: each output channel has one bias value (b[oc]) added before ReLU.
 *
 * apply_relu: 1 = apply ReLU after the weighted sum, 0 = skip it.
 *   Conv layers use ReLU; Dense2 does not (it goes through softmax instead).
 *
 * Why is this the bottleneck?
 *   Conv2 alone does: 6 × 49 × 64 × 32 × 3 × 3 = 54 million multiply-adds.
 *   The Nios V soft core has no hardware multiplier pipeline or SIMD, so this
 *   takes several seconds.  That is why PROCESSING takes so long on the board.
 */
static void conv2d_same(
    const float *in,  float *out,
    const float *w,   const float *b,
    int H, int W, int C_in, int C_out,
    int apply_relu)
{
    int oh, ow, oc, ic, kh, kw;

    /* Iterate over every output position (oh, ow) and every output channel oc */
    for (oh = 0; oh < H; oh++) {
        for (ow = 0; ow < W; ow++) {
            for (oc = 0; oc < C_out; oc++) {

                /* Start with the bias for this output channel */
                float acc = b[oc];

                /* Convolve: sum over all input channels and kernel positions */
                for (ic = 0; ic < C_in; ic++) {
                    for (kh = 0; kh < 3; kh++) {
                        /* ih = the input row that corresponds to kernel row kh.
                         * -1 offset implements the 'same' padding (kernel centre
                         *  at (oh, ow) means kernel top is at oh-1).
                         * Skip out-of-bounds rows (zero-padding implicitly). */
                        int ih = oh + kh - 1;
                        if (ih < 0 || ih >= H) continue;

                        for (kw = 0; kw < 3; kw++) {
                            int iw = ow + kw - 1;
                            if (iw < 0 || iw >= W) continue;

                            /* Input element at (ih, iw, ic) */
                            float x  = in[(ih * W + iw) * C_in + ic];
                            /* Weight for output channel oc, input channel ic,
                             * kernel position (kh, kw) */
                            float wt = w[((oc * C_in + ic) * 3 + kh) * 3 + kw];
                            acc += x * wt;
                        }
                    }
                }

                /* Apply ReLU if requested, then store in output tensor */
                float val = apply_relu ? relu(acc) : acc;
                out[(oh * W + ow) * C_out + oc] = val;
            }
        }
    }
}


/*
 * maxpool2d — 2×2 max pooling with stride 2 (no padding).
 *
 * What does max pooling do?
 *   It reduces the spatial dimensions by half.  For each non-overlapping 2×2
 *   block of input values, it outputs the largest (maximum) value.
 *
 * Why?
 *   1. It makes the representation smaller (fewer numbers → faster Dense layer).
 *   2. It makes the network "translation invariant" — a feature detected at
 *      position (4, 6) produces the same output if it shifts to (5, 6).
 *      This helps the CNN recognise "open sesame" whether you say it fast or
 *      slow, early or late in the 2-second window.
 *
 * Input:  [H][W][C]  — e.g. [13][99][32] after Conv1
 * Output: [H/2][W/2][C] — e.g. [6][49][32]
 *
 * Integer division floors: 13/2 = 6, 99/2 = 49.  The last row/column of the
 * input is dropped if the dimension is odd (no padding).
 */
static void maxpool2d(
    const float *in, float *out,
    int H, int W, int C)
{
    int oh, ow, oc;
    int OH = H / 2;   /* output height */
    int OW = W / 2;   /* output width  */

    for (oh = 0; oh < OH; oh++) {
        for (ow = 0; ow < OW; ow++) {
            for (oc = 0; oc < C; oc++) {
                /* Read the four input values in the 2×2 block at (2*oh, 2*ow) */
                float m = in[((oh*2)   * W + (ow*2)  ) * C + oc];  /* top-left     */
                float v;
                v = in[((oh*2)   * W + (ow*2+1)) * C + oc]; if (v > m) m = v; /* top-right    */
                v = in[((oh*2+1) * W + (ow*2)  ) * C + oc]; if (v > m) m = v; /* bottom-left  */
                v = in[((oh*2+1) * W + (ow*2+1)) * C + oc]; if (v > m) m = v; /* bottom-right */
                out[(oh * OW + ow) * C + oc] = m;  /* store the maximum */
            }
        }
    }
}


/*
 * dense — fully-connected (Dense) layer.
 *
 * What does a Dense layer do?
 *   Every input neuron connects to every output neuron.  Each output is a
 *   weighted sum of ALL inputs, plus a bias:
 *
 *     out[j] = bias[j] + Σ_i  in[i] × w[i][j]
 *
 *   The weights w[i][j] are the learned parameters.  After Conv/Pool extracts
 *   spatial features, the Dense layer combines them to make a final decision.
 *
 * Weight layout: [in_dim][out_dim] — matches Keras Dense layout directly.
 *
 * apply_relu: Dense1 uses ReLU, Dense2 does not (softmax handles it instead).
 *
 * Dropout (0.5) is applied during TRAINING only.  At inference time Dropout
 * is a no-op — Keras automatically disables it.  So we simply skip it.
 */
static void dense(
    const float *in, float *out,
    const float *w,  const float *b,
    int in_dim, int out_dim,
    int apply_relu)
{
    int i, j;
    for (j = 0; j < out_dim; j++) {
        float acc = b[j];   /* start with bias */
        for (i = 0; i < in_dim; i++)
            acc += in[i] * w[i * out_dim + j];  /* dot product: in · column j of w */
        out[j] = apply_relu ? relu(acc) : acc;
    }
}


/*
 * softmax — convert raw scores into probabilities.
 *
 * What is softmax?
 *   The Dense2 layer outputs two raw numbers (called "logits").  A higher
 *   number means the network is more confident about that class, but the
 *   numbers are not yet probabilities (they don't sum to 1).
 *
 *   Softmax transforms them:
 *     softmax(x[i]) = e^x[i] / Σ_j e^x[j]
 *
 *   After softmax, all values are in (0, 1) and they sum to exactly 1.0.
 *
 * Numerical stability trick:
 *   e^x can overflow float for large x.  We subtract the maximum value
 *   first: softmax(x[i] - max) gives the same result (the max cancels out
 *   in the division) but keeps the exponents in a safe range.
 *
 * After softmax:
 *   prob[0] = probability of class 0 ("not open sesame")
 *   prob[1] = probability of class 1 ("open sesame")
 *   prob[0] + prob[1] == 1.0
 */
static void softmax(float *x, int n)
{
    int i;

    /* Find the maximum value to subtract for numerical stability */
    float max_val = x[0];
    for (i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];

    /* Compute e^(x[i] - max) for all i and sum them up */
    float sum = 0.0f;
    for (i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }

    /* Normalise: divide each by the sum so they add to 1.0 */
    for (i = 0; i < n; i++) x[i] /= sum;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * run_inference — execute the full CNN forward pass
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Parameters:
 *   mfcc_in  — [N_MFCC][N_FRAMES] = [13][99] — output of compute_mfcc()
 *   prob_out — [2] — filled with softmax probabilities
 *
 * Data flow (using act_a / act_b ping-pong):
 *
 *   mfcc_in [13][99]    →  reshape to act_a [13][99][1]
 *   act_a               →  Conv1+ReLU   → act_b [13][99][32]
 *   act_b               →  MaxPool1     → act_a [6][49][32]
 *   act_a               →  Conv2+ReLU   → act_b [6][49][64]
 *   act_b               →  MaxPool2     → act_a [3][24][64]  (= Flatten input)
 *   act_a [4608]        →  Dense1+ReLU  → act_b [64]
 *   act_b [64]          →  Dense2       → prob_out [2]
 *   prob_out            →  Softmax      → prob_out [2]  (in-place)
 */
void run_inference(const float mfcc_in[N_MFCC][N_FRAMES], float prob_out[2])
{
    int h, w;

    /* ── Reshape mfcc_in [13][99] → act_a [13][99][1] ──────────────────
     * The conv2d_same function expects layout [H][W][C_in].
     * Our MFCC matrix is [N_MFCC][N_FRAMES] = [13][99] with no channel dim.
     * We add a dummy channel dimension (C_in = 1) by copying into act_a.
     * The data is identical; we just change how the index formula interprets it.
     * "(h * CONV1_W + w) * 1" = h * 99 + w, same as mfcc_in[h][w]. */
    for (h = 0; h < CONV1_H; h++)
        for (w = 0; w < CONV1_W; w++)
            act_a[(h * CONV1_W + w) * 1] = mfcc_in[h][w];

    /* ── Conv1: [13][99][1] → [13][99][32] + ReLU ───────────────────────
     * 32 filters of size 3×3 slide over the 13×99 spectrogram.
     * Each filter learns a different spectral-temporal pattern.
     * Input:  act_a (1287 floats), Output: act_b (41184 floats). */
    conv2d_same(act_a, act_b,
                &conv1_w[0][0][0][0], conv1_b,
                CONV1_H, CONV1_W, 1, CONV1_C, 1);   /* apply_relu = 1 */

    /* ── MaxPool1: [13][99][32] → [6][49][32] ───────────────────────────
     * Downsamples height 13→6, width 99→49. */
    maxpool2d(act_b, act_a, CONV1_H, CONV1_W, CONV1_C);

    /* ── Conv2: [6][49][32] → [6][49][64] + ReLU ────────────────────────
     * 64 filters learn higher-level patterns from the 32-channel feature maps. */
    conv2d_same(act_a, act_b,
                &conv2_w[0][0][0][0], conv2_b,
                POOL1_H, POOL1_W, CONV1_C, CONV2_C, 1);  /* apply_relu = 1 */

    /* ── MaxPool2: [6][49][64] → [3][24][64] ────────────────────────────
     * Downsamples again.  Output has 3*24*64 = 4608 values.
     * After this, the tensor is treated as a flat 1D vector (Flatten layer).
     * No explicit flatten step needed — the Dense function just reads it
     * linearly, which is equivalent. */
    maxpool2d(act_b, act_a, POOL1_H, POOL1_W, CONV2_C);

    /* ── Dense1: 4608 → 64 + ReLU ────────────────────────────────────────
     * Fully-connected layer.  Combines all 4608 features into 64 neurons.
     * Dropout(0.5) was applied during training but is a no-op at inference. */
    dense(act_a, act_b,
          &dense1_w[0][0], dense1_b,
          FLAT_SIZE, DENSE1_OUT, 1);  /* apply_relu = 1 */

    /* ── Dense2: 64 → 2 ──────────────────────────────────────────────────
     * Final classification layer.  Outputs 2 raw scores (logits).
     * No ReLU here — softmax follows immediately. */
    dense(act_b, prob_out,
          &dense2_w[0][0], dense2_b,
          DENSE1_OUT, DENSE2_OUT, 0);  /* apply_relu = 0 */

    /* ── Softmax: 2 scores → 2 probabilities ─────────────────────────────
     * Converts the raw logits to probabilities that sum to 1.0.
     * After this: prob_out[1] is P("open sesame"). */
    softmax(prob_out, DENSE2_OUT);
}

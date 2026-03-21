#include "mfcc.h"
#include <math.h>

/* Audio window length: frame 0 starts at 0, last frame starts at (N_FRAMES-1)*HOP_LENGTH */
#define AUDIO_WINDOW_LEN  (N_FFT + HOP_LENGTH * (N_FRAMES - 1))   /* 15880 */

/* power_to_db parameters — must match librosa defaults */
#define AMIN_DB   1e-10f
#define TOP_DB    80.0f

/* Static scratch buffers — never on the stack */
static float s_log_mel[N_MELS][N_FRAMES];   /* 26*99 floats — log-mel energies */
static float s_frame[N_FFT];                 /* 200 floats   — windowed frame   */
static float s_power[N_BINS];               /* 101 floats   — power spectrum   */
static float s_mel[N_MELS];                 /* 26  floats   — mel energies     */

void compute_mfcc(const float *audio, float mfcc_out[N_MFCC][N_FRAMES])
{
    int t, n, k, m, i;

    /* ── Pass 1: compute log-mel for every frame ─────────────────────────── */
    for (t = 0; t < N_FRAMES; t++) {
        int start = t * HOP_LENGTH;

        /* Apply Hann window */
        for (n = 0; n < N_FFT; n++)
            s_frame[n] = audio[start + n] * hann_win[n];

        /* DFT: only positive frequencies (k = 0 .. N_BINS-1 = 101 bins)
         * cos_table[k][n] = cos(2*pi*k*n/N_FFT)
         * sin_table[k][n] = sin(2*pi*k*n/N_FFT)
         * Power spectrum = Re^2 + Im^2 (no sqrt needed for mel filterbank) */
        for (k = 0; k < N_BINS; k++) {
            float re = 0.0f, im = 0.0f;
            const float *ck = cos_table[k];
            const float *sk = sin_table[k];
            for (n = 0; n < N_FFT; n++) {
                re += s_frame[n] * ck[n];
                im += s_frame[n] * sk[n];
            }
            s_power[k] = re * re + im * im;
        }

        /* Mel filterbank: mel_fb[m][k] weights — sparse triangular filters */
        for (m = 0; m < N_MELS; m++) {
            float e = 0.0f;
            const float *fb = mel_fb[m];
            for (k = 0; k < N_BINS; k++)
                e += fb[k] * s_power[k];
            s_mel[m] = e;
        }

        /* power_to_db: 10 * log10(max(AMIN, mel_energy))
         * Matches librosa.power_to_db(S, ref=1.0, amin=1e-10) */
        for (m = 0; m < N_MELS; m++) {
            float val = (s_mel[m] < AMIN_DB) ? AMIN_DB : s_mel[m];
            s_log_mel[m][t] = 10.0f * log10f(val);
        }
    }

    /* ── Pass 2: apply top_db clipping (global max over all frames/bands) ── */
    float global_max = s_log_mel[0][0];
    for (m = 0; m < N_MELS; m++)
        for (t = 0; t < N_FRAMES; t++)
            if (s_log_mel[m][t] > global_max)
                global_max = s_log_mel[m][t];

    float floor_val = global_max - TOP_DB;
    for (m = 0; m < N_MELS; m++)
        for (t = 0; t < N_FRAMES; t++)
            if (s_log_mel[m][t] < floor_val)
                s_log_mel[m][t] = floor_val;

    /* ── Pass 3: DCT-II via precomputed matrix + normalize ──────────────── */
    /* dct_mat[i][m] = ortho-normalized DCT-II basis vector
     * mfcc_out[i][t] = sum_m dct_mat[i][m] * s_log_mel[m][t]
     * Then apply MFCC_MEAN / MFCC_STD to match training normalization.      */
    for (i = 0; i < N_MFCC; i++) {
        for (t = 0; t < N_FRAMES; t++) {
            float val = 0.0f;
            for (m = 0; m < N_MELS; m++)
                val += dct_mat[i][m] * s_log_mel[m][t];
            mfcc_out[i][t] = (val - MFCC_MEAN) / MFCC_STD;
        }
    }
}

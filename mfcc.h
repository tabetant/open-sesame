#ifndef MFCC_H
#define MFCC_H

#include "output/model_data.h"

/*
 * compute_mfcc()
 *
 * Extracts 13 MFCCs over 99 frames from 2 seconds of 8 kHz audio.
 *
 * Exactly matches librosa.feature.mfcc with:
 *   sr=8000, n_mfcc=13, n_fft=200, hop_length=160, n_mels=26, center=False
 *   followed by power_to_db normalization and the project's MFCC_MEAN/MFCC_STD.
 *
 * audio    : N_FFT + HOP_LENGTH*(N_FRAMES-1) = 15880 float samples, [-1.0, 1.0]
 * mfcc_out : [N_MFCC][N_FRAMES] = [13][99], normalized and ready for inference
 */
void compute_mfcc(const float *audio, float mfcc_out[N_MFCC][N_FRAMES]);

#endif /* MFCC_H */

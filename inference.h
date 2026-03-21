#ifndef INFERENCE_H
#define INFERENCE_H

#include "output/model_data.h"

/*
 * run_inference()
 *
 * Runs the CNN forward pass on normalized MFCC features.
 *
 * mfcc_in  : [N_MFCC][N_FRAMES] = [13][99], already normalized (from compute_mfcc)
 * prob_out : [2] — prob_out[0] = P(negative), prob_out[1] = P(positive)
 *
 * If prob_out[1] > 0.9, the model heard "open sesame".
 */
void run_inference(const float mfcc_in[N_MFCC][N_FRAMES], float prob_out[2]);

#endif /* INFERENCE_H */

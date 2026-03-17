#include "address_map.h"
#define BUFFER_SIZE 8000 // 8000 for 1 second of audio at 8kHz
#define SAMPLE_RATE 8000

struct AUDIO_T
{
    volatile unsigned int control;
    volatile unsigned char rarc;
    volatile unsigned char ralc;
    volatile unsigned char wsrc;
    volatile unsigned char wslc;
    volatile int ldata;
    volatile int rdata;
};

// will hold the samples we are giving over to neural network
// for processing, it can only handle frames, not individual samples
int audio_buffer[BUFFER_SIZE];

int main()
{
    // initialize audio buffer index and pointer to audio device
    int buf_index = 0;
    struct AUDIO_T *audio_ptr = (struct AUDIO_T *)AUDIO_BASE;

    // circular loop
    while (1)
    {
        if (audio_ptr->rarc)
        {
            int left = audio_ptr->ldata;
            int right = audio_ptr->rdata;

            // store sample in buffer
            audio_buffer[buf_index++] = left;

            // once filled up, pass to neural network
            // neural network code for later
            if (buf_index >= BUFFER_SIZE)
            {
                buf_index = 0;
                // process window here
            }

            // TESTING: write back to speakers to hear sound
            audio_ptr->ldata = left;
            audio_ptr->rdata = right;
        }
    }
    return 0;
}
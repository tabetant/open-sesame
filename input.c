#include "address_map.h"

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

int main()
{
    struct AUDIO_T *audio_ptr = (struct AUDIO_T *)AUDIO_BASE;
    while (1)
    {
        if (audio_ptr->rarc)
        {
            int left = audio_ptr->ldata;
            int right = audio_ptr->rdata;
            audio_ptr->ldata = left;
            audio_ptr->rdata = right;
        }
    }
    return 0;
}
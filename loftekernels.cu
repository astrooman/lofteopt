#include <iostream>
#include <fstream>

#include <stdio.h>

#include "errors.hpp"

#define ACC 1024
#define TIMEAVG 8
#define TIMESCALE 0.125
#define FFTOUT 513
#define FFTUSE 512

using std::cout;
using std::endl;

int power2factor(unsigned int inbytes) {
    if ((inbytes % 2) != 0)
        return 1;      // don't even  bother with odd numbers

    int factor = 4;

    while ((inbytes % factor) == 0) {
        factor *= 2;
    }

    return factor / 2;
};

__device__ float fftfactor = 1.0;

__constant__ unsigned char kMask[] = {0x03, 0x0C, 0x30, 0xC0};

__global__ void UnpackKernel(unsigned char **in, float **out, int nopols, int bytesperthread, int rem, size_t samples, int unpack)
{
    int idx = blockIdx.x * blockDim.x * bytesperthread + threadIdx.x * bytesperthread;

    if (idx < samples) {
        for (int ipol = 0; ipol < nopols; ipol++) {
            for (int ibyte = 0; ibyte < bytesperthread; ibyte++) {
                for (int isamp = 0; isamp < unpack; isamp++) {
                    out[ipol][idx * unpack + ibyte * unpack + isamp] = static_cast<float>(static_cast<short>((in[ipol][idx + ibyte] & kMask[isamp]) >> ( 2 * isamp)));
                }
            }
        }
    }
}

__global__ void UnpackKernelOpt(unsigned char **in, float **out, int nopols, int bytesperthread, int rem, size_t samples, int unpack)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < samples) {
	for (int ipol = 0; ipol < 2; ++ipol) {
	    unsigned char inval = in[ipol][idx];
	    out[ipol][idx * unpack] = static_cast<float>(static_cast<short>((inval & 0x03)));
            out[ipol][idx * unpack + 1] = static_cast<float>(static_cast<short>((inval & 0x0C >> 2)));
            out[ipol][idx * unpack + 2] = static_cast<float>(static_cast<short>((inval & 0x30 >> 4)));
            out[ipol][idx * unpack + 3] = static_cast<float>(static_cast<short>((inval & 0xC0 >> 6)));
        }
    }
}

// NOTE: This kernel assumes that 2-bit data sampling is used
__global__ void UnpackKernelOpt2(unsigned char **in, float **out, int nopols, int perblock, size_t samples)
{

    int idx = blockIdx.x * blockDim.x * perblock + threadIdx.x;

    for (int isamp = 0; isamp < perblock; ++isamp) {

        idx += blockDim.x;

        if (idx < samples) {
            for (int ipol = 0; ipol < 2; ++ipol) {
                unsigned char inval = in[ipol][idx];
		// NOTE: Strided access - the worst of its kind - might need to used shared memory here
                out[ipol][idx * 4] = static_cast<float>(static_cast<short>((inval & 0x03))); 
                out[ipol][idx * 4 + 1] = static_cast<float>(static_cast<short>((inval & 0x0C >> 2)));
                out[ipol][idx * 4 + 2] = static_cast<float>(static_cast<short>((inval & 0x30 >> 4)));
                out[ipol][idx * 4 + 3] = static_cast<float>(static_cast<short>((inval & 0xC0 >> 6)));
            }
        }
    }
}

__global__ void UnpackKernelOpt3(unsigned char **in, float **out, int nopols, int perblock, size_t samples)
{

    int idx = blockIdx.x * blockDim.x * perblock + threadIdx.x;
    int tmod = threadIdx.x % 4;

    // NOTE: Each thread can store one value
    __shared__ unsigned char incoming[1024];

    int outidx = blockIdx.x * blockDim.x * perblock * 4;

    for (int isamp = 0; isamp < perblock; ++isamp) {
        if (idx < samples) {
            for (int ipol = 0; ipol < 2; ++ipol) {
                incoming[threadIdx.x] = in[ipol][idx];
                __syncthreads();
                int outidx2 = outidx + threadIdx.x;
		for (int ichunk = 0; ichunk < 4; ++ichunk) {
                    int inidx = threadIdx.x / 4 + ichunk * 256;
                    unsigned char inval = incoming[inidx];
                    out[ipol][outidx2] = static_cast<float>(static_cast<short>(((inval & kMask[tmod]) >> (2 * tmod))));
                    outidx2 += 1024;
                }
            }
        }
        idx += blockDim.x;
        outidx += blockDim.x * 4;
    }
}


__global__ void PowerScaleKernel(cufftComplex **in, unsigned char **out, int avgfreq, int avgtime, int nchans, int outsampperblock,
                                    int inskip, int nogulps, int gulpsize, int extra, unsigned int framet,
                                    unsigned int perframe)
{
    // NOTE: nchans is the number of frequency channels AFTER the averaging
    unsigned int inidx = 0;
    unsigned int outidx = 0;
    unsigned int filtimeidx = 0;
    unsigned int filfullidx = 0;

    for (int ichunk = 0; ichunk < outsampperblock; ichunk++) {
        filtimeidx = framet * perframe + blockIdx.x * outsampperblock + ichunk;
        filfullidx = (filtimeidx % (nogulps * gulpsize)) * nchans;
        outidx = filfullidx + threadIdx.x;
        for (int isamp = 0; isamp < avgtime; isamp++) {
            for (int ifreq = 0; ifreq < avgfreq; ifreq++) {
                //inidx = inskip + blockIdx.x * avgtime * nchans * outsampperblock + ichunk * nchans * avgtime + isamp * nchans + threadIdx.x * avgfreq + ifreq;
		//inidx = inskip + blockIdx.x * outsampperblock * avgtime * nchans * avgfreq + ichunk * avgtime * nchans * avgfreq + isamp * nchans * avgfreq  + threadIdx.x * avgfreq + ifreq;
                inidx = inskip + blockIdx.x * outsampperblock * avgtime * (nchans + 1) * avgfreq + ichunk * avgtime * (nchans + 1) * avgfreq + isamp * (nchans + 1) * avgfreq  + threadIdx.x * avgfreq + ifreq + 1;
                out[0][outidx] += in[0][inidx].x * in[0][inidx].x + in[0][inidx].y * in[0][inidx].y + in[1][inidx].x * in[1][inidx].x + in[1][inidx].y * in[1][inidx].y;
                //out[1][outidx] += in[0][inidx].x * in[0][inidx].x + in[0][inidx].y * in[0][inidx].y + in[1][inidx].x * in[1][inidx].x + in[1][inidx].y * in[1][inidx].y;
                //out[2][outidx] += 2.0f * in[0][inidx].x * in[1][inidx].x + 2.0f * in[0][inidx].y * in[1][inidx].y;
                //out[3][outidx] += 2.0f * in[0][inidx].x * in[1][inidx].y + 2.0f * in[0][inidx].y * in[1][inidx].x;
            }
        }

        if (filfullidx < extra) {
            out[0][outidx + nogulps * gulpsize * nchans] = out[0][outidx];
            //out[1][outidx + nogulps * gulpsize * nchans] = out[1][outidx];
            //out[2][outidx + nogulps * gulpsize * nchans] = out[2][outidx];
            //out[3][outidx + nogulps * gulpsize * nchans] = out[3][outidx];
        }
    }
}

// NOTE: This kernel does not do any frequency averaging - this can really be changed with the size of the FFT
__global__ void PowerScaleKernelOpt(cufftComplex **in, unsigned char **out, int avgfreq, int avgtime, int nchans, int perblock,
                                    int inskip, int nogulps, int gulpsize, int extra, unsigned int framet,
                                    unsigned int perframe)
{
    // NOTE: framet should start at 0 and increase by accumulate every time this kernel is called
    // NOTE: REALLY make sure it starts at 0
    unsigned int filtime = framet / ACC * gridDim.x * perblock + blockIdx.x * perblock;  
    unsigned int filidx;
    // NOTE: Row for every channel and column for every time sample to average over + padding
    __shared__ cufftComplex pol[FFTUSE][TIMEAVG];

    int inidx = blockIdx.x * FFTOUT * perblock * TIMEAVG + threadIdx.x + 1;
   
    float outvalue = 0.0f;
    cufftComplex polval;
 
    for (int isamp = 0; isamp < perblock; ++isamp) {
        
        // NOTE: Read the data from the incoming array
        for (int ipol = 0; ipol < 2; ++ipol) {
            for (int iavg = 0; iavg < TIMEAVG; ++iavg) {
                // NOTE: threadIdx.x + 1 skips the DC compoment - is it the correct thing to do?
                pol[threadIdx.x][iavg] = in[ipol][inidx + iavg * FFTOUT]; 
            }

            for (int itime = 0; itime < TIMEAVG; itime++) {
                polval = pol[threadIdx.x][itime];
                outvalue += polval.x * polval.x + polval.y * polval.y; 
            } 
    
        }

        filidx = filtime % (nogulps * gulpsize);
        int outidx = filidx * FFTUSE + threadIdx.x;

        out[0][outidx] = outvalue;
        // NOTE: Save to the extra part of the buffer
        if (filidx < extra) {
            out[0][outidx + nogulps * gulpsize * FFTUSE] = outvalue;
        }
        inidx += FFTOUT * TIMEAVG;
        filtime++;
        outvalue = 0.0;
    }
}

__global__ void PowerScaleKernelOpt2(cufftComplex **in, unsigned char **out, int avgfreq, int avgtime, int nchans, int perblock,
                                    int inskip, int nogulps, int gulpsize, int extra, unsigned int framet) {
    // NOTE: framet should start at 0 and increase by accumulate every time this kernel is called
    // NOTE: REALLY make sure it starts at 0
    // NOTE: I'M SERIOUS - FRAME TIME CALCULATIONS ARE BASED ON THIS ASSUMPTION
    unsigned int filtime = framet / ACC * gridDim.x * perblock + blockIdx.x * perblock;
    unsigned int filidx;
    unsigned int outidx;
    int inidx = blockIdx.x * perblock * TIMEAVG * FFTOUT + threadIdx.x + 1;

    float outvalue = 0.0f;
    cufftComplex polval;

    for (int isamp = 0; isamp < perblock; ++isamp) {

        // NOTE: Read the data from the incoming array
        for (int ipol = 0; ipol < 2; ++ipol) {
            for (int iavg = 0; iavg < TIMEAVG; iavg++) {
                polval = in[ipol][inidx + iavg * FFTOUT];
                outvalue += polval.x * polval.x + polval.y * polval.y;
            }

        }

        filidx = filtime % (nogulps * gulpsize);
        outidx = filidx * FFTUSE + threadIdx.x;

        outvalue *= TIMESCALE;

        out[0][outidx] = outvalue;
        // NOTE: Save to the extra part of the buffer
        if (filidx < extra) {
            out[0][outidx + nogulps * gulpsize * FFTUSE] = outvalue;
        }
        inidx += FFTOUT * TIMEAVG;
        filtime++;
        outvalue = 0.0;
    }
}

int main(int argc, char*argv[]) {

    int accumulate = 4000;
    int vdiflen = 8000;
    int inbits = 2;
    int nopols = 2;
    int nostokes = 4;

    unsigned char *rawdata = new unsigned char[accumulate * vdiflen];

    for (int isamp = 0; isamp < accumulate * vdiflen; ++isamp) {
        rawdata[isamp] = isamp % 256;
    }

    // NOTE: BUffers divided per polarisations
    unsigned char **hrawdata = new unsigned char*[nopols];
    float **hunpacked = new float*[nopols];
    cufftComplex **hffted = new cufftComplex*[nopols];
    for (int ipol = 0; ipol < nopols; ++ipol) {
        cudaCheckError(cudaMalloc((void**)&hrawdata[ipol], accumulate * vdiflen * sizeof(unsigned char)));
        cudaCheckError(cudaMemcpy(hrawdata[ipol], rawdata, accumulate * vdiflen * sizeof(unsigned char), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMalloc((void**)&hunpacked[ipol], accumulate * vdiflen * 4 * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&hffted[ipol], accumulate * vdiflen * 4 * sizeof(cufftComplex)));
    }

    unsigned char **drawdata;
    cudaCheckError(cudaMalloc((void**)&drawdata, nopols * sizeof(unsigned char*)));
    cudaCheckError(cudaMemcpy(drawdata, hrawdata, nopols * sizeof(unsigned char*), cudaMemcpyHostToDevice));
    float **dunpacked;
    cudaCheckError(cudaMalloc((void**)&dunpacked, nopols * sizeof(float*)));
    cudaCheckError(cudaMemcpy(dunpacked, hunpacked, nopols * sizeof(float*), cudaMemcpyHostToDevice));
    cufftComplex **dffted;
    cudaCheckError(cudaMalloc((void**)&dffted, nopols * sizeof(cufftComplex*)));
    cudaCheckError(cudaMemcpy(dffted, hffted, nopols * sizeof(cufftComplex*), cudaMemcpyHostToDevice));

    int nokernels = 3;  // unpack, power and scale
    unsigned int *cudathreads = new unsigned int[nokernels];
    unsigned int *cudablocks = new unsigned int[nokernels];

    // currently limit set to 1024, but can be lowered down, depending on the results of the tests
    int sampperthread = min(power2factor(accumulate * vdiflen), 1024);
    int needthreads = accumulate * vdiflen / sampperthread;
    cudathreads[0] = min(needthreads, 1024);
    int needblocks = (needthreads - 1) / cudathreads[0] + 1;

    cudablocks[0] = min(needblocks, 65536);
    int rem = needthreads - cudablocks[0] * cudathreads[0];

    int avgfreq = 1;
    int avgtime = 1;
    int filchans = 128 / avgfreq;
    int fftsize = 256;

    int perblock = 100;        // the number of OUTPUT time samples (after averaging) per block
    // NOTE: Have to be very careful with this number as vdiflen is not a power of 2
    // This will cause problems when accumulate_ * 8 / inbits is less than 1 / (avgtime_ * perblock_ * fftsize_[0]
    // This will give a non-integer number of blocks
    cudablocks[1] = accumulate * vdiflen * 8 / inbits / avgtime / perblock / fftsize;
    cudathreads[1] = filchans;        // each thread in the block will output a single AVERAGED frequency channel

    // NOTE: This should only be used for debug purposes
    cout << "Unpack kernel grid: " << cudablocks[0] << " blocks and " << cudathreads[0] << " threads" <<endl;
    cout << "Power kernel grid: " << cudablocks[1] << " blocks and " << cudathreads[1] << " threads" <<endl;

    unsigned char **hfilbuffer = new unsigned char*[nostokes];
    for (int istoke = 0; istoke < nostokes; ++istoke) {
        cudaCheckError(cudaMalloc((void**)&hfilbuffer[istoke], 131072 * 128 * sizeof(unsigned char)))
    }

    cout << "Allocated the filterbank buffer memory" << endl;

    unsigned char **dfilbuffer;
    cudaCheckError(cudaMalloc((void**)&dfilbuffer, nostokes * sizeof(unsigned char*)));
    cudaCheckError(cudaMemcpy(dfilbuffer, hfilbuffer, nostokes * sizeof(unsigned char*), cudaMemcpyHostToDevice));

    unsigned int perframe = vdiflen * 8 / inbits / 256 / avgtime;

    // NOTE: These kernels have to be as close to the original as possible
    // NOTE: Sampperthread is confusing - is is really the number of incoming bytes processed per thread
    //UnpackKernel<<<cudablocks[0], cudathreads[0], 0, 0>>>(drawdata, dunpacked, nopols, sampperthread, rem, accumulate * vdiflen, 8 / inbits);
    //cudaDeviceSynchronize();
    //cudaCheckError(cudaGetLastError());
    //cout << "Unpacked the data" << endl;
    //PowerScaleKernel<<<cudablocks[1], cudathreads[1], 0, 0>>>(dffted, dfilbuffer, avgfreq, avgtime, filchans, perblock, 0, 1, 131072, 0, 0, perframe);
    //cudaDeviceSynchronize();
    //cudaCheckError(cudaGetLastError());
    //cout << "Got the power" << endl;
    // NOTE: That version calls 8000 blocks for accumulate of 4000 - more than I'm comfortable with
    //UnpackKernelOpt<<<accumulate * vdiflen / 1024, 1024, 0, 0>>>(drawdata, dunpacked, nopols, sampperthread, rem, accumulate * vdiflen, 8 / inbits);
    //cudaDeviceSynchronize();
    //cudaCheckError(cudaGetLastError());
    //UnpackKernelOpt2<<<accumulate * vdiflen / 1024 / (2 * perblock), 1024, 0, 0>>>(drawdata, dunpacked, nopols, 2 * perblock, accumulate * vdiflen);
    //cudaDeviceSynchronize();
    //cudaCheckError(cudaGetLastError());
    UnpackKernelOpt3<<<50, 1024, 0, 0>>>(drawdata, dunpacked, nopols, 625, accumulate * vdiflen);
    cudaDeviceSynchronize();
    cudaCheckError(cudaGetLastError());
    PowerScaleKernelOpt<<<25, 512, 0, 0>>>(dffted, dfilbuffer, avgfreq, avgtime, filchans, 625, 0, 1, 131072, 0, 0, perframe);
    cudaDeviceSynchronize();
    cudaCheckError(cudaGetLastError());
    PowerScaleKernelOpt2<<<25, 512, 0, 0>>>(dffted, dfilbuffer, avgfreq, avgtime, filchans, 625, 0, 1, 131072, 0, 0);
    cudaDeviceSynchronize();
    cudaCheckError(cudaGetLastError());
}

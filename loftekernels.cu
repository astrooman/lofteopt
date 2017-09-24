#include <iostream>

#include <stdio.h>

#include "errors.hpp"

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
                out[1][outidx] += in[0][inidx].x * in[0][inidx].x + in[0][inidx].y * in[0][inidx].y + in[1][inidx].x * in[1][inidx].x + in[1][inidx].y * in[1][inidx].y;
                out[2][outidx] += 2.0f * in[0][inidx].x * in[1][inidx].x + 2.0f * in[0][inidx].y * in[1][inidx].y;
                out[3][outidx] += 2.0f * in[0][inidx].x * in[1][inidx].y + 2.0f * in[0][inidx].y * in[1][inidx].x;
            }
        }

        if (filfullidx < extra) {
            out[0][outidx + nogulps * gulpsize * nchans] = out[0][outidx];
            out[1][outidx + nogulps * gulpsize * nchans] = out[1][outidx];
            out[2][outidx + nogulps * gulpsize * nchans] = out[2][outidx];
            out[3][outidx + nogulps * gulpsize * nchans] = out[3][outidx];
        }
    }
}

int main(int argc, char*argv[]) {

    int accumulate = 1024;
    int vdiflen = 8000;
    int inbits = 2;
    int nopols = 2;
    int nostokes = 4;

    // NOTE: Single buffer for both polarisations
    unsigned char **rawdata;
    cudaCheckError(cudaMalloc((void**)&rawdata, accumulate * vdiflen * nopols * sizeof(unsigned char)));
    float *unpacked;
    cudaCheckError(cudaMalloc((void**)&unpacked, accumulate * vdiflen * nopols * 4 * sizeof(float)));

    // NOTE: BUffers divided per polarisations
    unsigned char **hrawdata = new unsigned char*[nopols];
    float **hunpacked = new float*[nopols];
    cufftComplex **hffted = new cufftComplex*[nopols];
    for (int ipol = 0; ipol < nopols; ++ipol) {
        cudaCheckError(cudaMalloc((void**)&hrawdata[ipol], accumulate * vdiflen * sizeof(unsigned char)));
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

    unsigned char **dfilbuffer;
    cudaCheckError(cudaMalloc((void**)&dfilbuffer, nostokes * sizeof(unsigned char*)));
    cudaCheckError(cudaMemcpy(dfilbuffer, hfilbuffer, nostokes * sizeof(unsigned char*), cudaMemcpyHostToDevice));

    unsigned int perframe = vdiflen * 8 / inbits / 256 / avgtime;

    // NOTE: These kernels have to be as close to the original as possible
    UnpackKernel<<<cudablocks[0], cudathreads[0], 0, 0>>>(drawdata, dunpacked, nopols, sampperthread, rem, accumulate * vdiflen, 8 / inbits);
    cudaCheckError(cudaGetLastError());
    PowerScaleKernel<<<cudablocks[1], cudathreads[1], 0, 0>>>(dffted, dfilbuffer, avgfreq, avgtime, filchans, perblock, 0, 1, 131072, 0, 0, perframe);
    cudaCheckError(cudaGetLastError());
}

#include "kernel.h"
#define TX 16
#define TY 16
#include <curand.h>
#include <curand_kernel.h>
#define PI 3.141592654f
#define TPB 256 // reduction

__device__
float findX(int idx, int w) {
    return ((idx - idx/w * w) * SCALEX / (w / 2) - SCALEX);
}
__device__
float findY(int idx, int h) {
    return  (SCALEY - idx/h * SCALEY / (h / 2));
}

int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
float clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__device__ 
unsigned char value(float n1, float n2, int hue) {
    if (hue > 360)      hue -= 360;
    else if (hue < 0)   hue += 360;

    if (hue < 60)
        return (unsigned char)(255 * (n1 + (n2 - n1) * hue / 60));
    if (hue < 180)
        return (unsigned char)(255 * n2);
    if (hue < 240)
        return (unsigned char)(255 * (n1 + (n2 - n1) * (240 - hue) / 60));
    return (unsigned char)(255 * n1);
}

__device__
int idxClip(int idx, int idxMax) {
    return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int col, int row, int width, int height) {
    return idxClip(col, width) + idxClip(row, height) * width;
}


__device__
int getBin(float varX, float varY, int w) {
    int x = int(varX / (float(SCALEX) / float(w / 2))) + w / 2;
    int y = w / 2 - int(varY / (float(SCALEY) / float(w / 2)));
    return  x + y * w;
}

//////////////////////////////////////////////////////////////////
__global__
void ReductionKernel(float3* d_particals,float * d_doubderv, float* d_dnscell) {
    __shared__ float cache1[SQRTNUM];
    __shared__ float cache2[SQRTNUM];
    for (int j = 0; j < SQRTNUM * SQRTNUM; j++) {
        unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        float xi = d_particals[j].x, pi = d_particals[j].y;
        
        float temp1 = 0.0;
        float temp2 = 0.0;
        while (index < NUM) {
            float x = d_particals[index].x, p = d_particals[index].y, time = d_particals[index].z;
            if (time >= 0) {
                temp1 += 1 / (2 * PI * HX * HP * NUM) * (pow(p - pi, 2) - HP * HP) / (pow(HP, 4)) *
                    expf(-pow(x - xi, 2) / (2 * HX * HX) - pow(p - pi, 2) / (2 * HP * HP));
                temp2 += 1 / (2 * PI * HX * HP * NUM) *
                    expf(-pow(x - xi, 2) / (2 * HX * HX) - pow(p - pi, 2) / (2 * HP * HP));
                index += stride;
            }
            else{
                temp1 += 0;
                temp2 += 0;
                index += stride;
            }
        }

        cache1[threadIdx.x] = temp1;
        cache2[threadIdx.x] = temp2;
        __syncthreads();

        // reduction
        unsigned int i = blockDim.x / 2;
        while (i != 0) {
            if (threadIdx.x < i) {
                cache1[threadIdx.x] += cache1[threadIdx.x + i];
                cache2[threadIdx.x] += cache2[threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }


        if (threadIdx.x == 0) {
            atomicAdd(&d_doubderv[j], cache1[0]);
            atomicAdd(&d_dnscell[j], cache2[0]);
        }
    }
}

__global__
void clearDensityAndOutArrKernel(uchar4* d_out, float* d_density, int w, int h) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = flatten(col, row, w, h);

    d_density[idx] = 0;
    d_out[idx].x = 0;
    d_out[idx].y = 0;
    d_out[idx].z = 0;
    d_out[idx].w = 255;
}
__global__
void plotKernel(uchar4* d_out, float* d_density, int w, int h) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = flatten(col, row, w, h);
    if ((col >= w) || (row >= h)) return;
        float l = d_density[idx];
        float s = 1;
        int hc = (180 + (int)(360.0f * d_density[idx] * 1.5 )) % 360;
        float m1, m2;

        if (l <= 5.0f)
            m2 = l * (1 + s);
        else
            m2 = l + s - l * s;
        m1 = 2 * l - m2;
        d_out[idx].x = 255 * d_density[idx];
        d_out[idx].y = 0;
        d_out[idx].z = 0;
        d_out[idx].w = 255;
        //d_out[idx].x = value(m1, m2, hc + 120);;
        //d_out[idx].y = value(m1, m2, hc);
        //d_out[idx].z = value(m1, m2, hc - 120);
        //d_out[idx].w = 255;

        if (col == h / 2 || row == w / 2) {
            d_out[idx].x = 255;
            d_out[idx].y = 255;
            d_out[idx].z = 255;
            d_out[idx].w = 255;
        }
}

__global__
void makeHistKernel(float3* d_particals, float* d_density, int sqrtnum, int wight) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = flatten(col, row, sqrtnum, sqrtnum);
    if ((col >= sqrtnum) || (row >= sqrtnum)) return;
    float x = d_particals[idx].x, p = d_particals[idx].y, time = d_particals[idx].z;
    
    if (time > 0) {
        int mapIdx = getBin(x, p, wight);
        d_density[mapIdx] = 1;
    }
}

__global__
void makeHistKernelColor(float3* d_particals, float* d_density, int w, int h) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = flatten(col, row, w, h);

    if ((col >= w) || (row >= h)) return;

    float sum = 0.0f;
    float x = findX(idx, w), y = findY(idx, h);

    for (int i = 0; i < NUM; i++) {
        float xi = d_particals[i].x, yi = d_particals[i].y;
        sum += 1 / (2 * PI * HX * HP) * expf(-(x - xi) * (x - xi) / (2 * HX * HX) - (y - yi) * (y - yi) / (2 * HP * HP));
    }
    d_density[idx] = sum / float(NUM);
}

__global__
void timeNextKernel(float3 * d_particals, float * d_doubderv, float* d_dnscell,  int sqrtnum) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = flatten(col, row, sqrtnum, sqrtnum);
    if ((col >= sqrtnum) || (row >= sqrtnum)) return;
    float x = d_particals[idx].x, p = d_particals[idx].y, time = d_particals[idx].z;
    float potderv = APOT * x - BPOT * x * x; // potential derivative
    float pot3derv = -2 * BPOT; // potential derivative

    float doubderv = d_doubderv[idx]; // double dot density
    float dnscell = d_dnscell[idx]; // density value in cell


    float deltaX = STEPTIME * (p / MASS);
    float deltaP = STEPTIME * (-potderv + pot3derv / (4 * 6) * doubderv / dnscell);
    if (deltaX + x > SCALEX || deltaX + x < -SCALEX || deltaP + p > SCALEY || deltaP + p < -SCALEY || time < 0) {
        d_particals[idx].x = -SCALEX;
        d_particals[idx].y = -SCALEY;
        d_particals[idx].z = -1;
    }
    else {
        d_particals[idx].x += deltaX;
        d_particals[idx].y += deltaP;
        d_particals[idx].z += STEPTIME;
    }
}

//////////////////////////////////////////////////////////////////

void initialConditions(uchar4 * d_out, float3 * d_particals, float * d_density, int w, int h) {
    float3* initialRandomValues = new float3[NUM];
    FILE* infile = fopen("randForCubic2.txt", "r");
    for (int i = 0; i < NUM; i++) {
        if (fscanf(infile, "%f \t %f\t %f \n", &initialRandomValues[i].x, &initialRandomValues[i].y, &initialRandomValues[i].z) == EOF) break;
    }
    fclose(infile);

    cudaMemcpy(d_particals, initialRandomValues, NUM * sizeof(float3), cudaMemcpyHostToDevice);

    const dim3 blockSizePart(TX, TY);
    const dim3 gridSizePart(divUp(SQRTNUM, 4), divUp(SQRTNUM, 4));
    makeHistKernel <<<gridSizePart, blockSizePart >>> (d_particals, d_density, SQRTNUM, w);

    //const dim3 blockSize(TX, TY);
    //const dim3 gridSize(divUp(w, TX), divUp(h, TY));
    //makeHistKernelColor << <gridSize, blockSize >> > (d_particals, d_density, w, h);
}

void kernelLauncher(uchar4 * d_out, float3 * d_particals, float * d_density, float * d_doubderv, float * d_dnscell, int w, int h) {

    const dim3 blockSize(TX, TY);
    const dim3 gridSize(divUp(w, TX), divUp(h, TY));
    const dim3 blockSizePart(TX, TY);
    const dim3 gridSizePart(divUp(SQRTNUM, TX), divUp(SQRTNUM, TX));

    //Reduction
    dim3 gridSizeReduction = 16;
    dim3 blockSizeReduction = 4 * 16;
    cudaMemset(d_doubderv, 0.0, NUM * sizeof(float));
    cudaMemset(d_dnscell, 0.0, NUM * sizeof(float));
    ReductionKernel<<<gridSizeReduction, blockSizeReduction >>> (d_particals, d_doubderv, d_dnscell);

    timeNextKernel<<<gridSizePart, blockSizePart>>>(d_particals,d_doubderv, d_dnscell, SQRTNUM);
    clearDensityAndOutArrKernel <<<gridSize, blockSize >>> (d_out, d_density, w, h);
    makeHistKernel<<<gridSizePart, blockSizePart >>> (d_particals, d_density, SQRTNUM, w);
    //makeHistKernelColor << <gridSize, blockSize >> > (d_particals, d_density, w, h);
    plotKernel <<<gridSize, blockSize >>> (d_out, d_density, w, h);
}

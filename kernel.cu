#include "kernel.h"
#define TX 32
#define TY 32
#include <curand.h>
#include <curand_kernel.h>


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

__global__
void setupKernel(curandState* state, unsigned long seed, int w, int h)
{

    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = flatten(col, row, w, h);
    curand_init(seed, idx, 0, &state[idx]);
}

__device__
int getBin(float varX, float varY, int w) {
    int x = int(varX / (float(SCALE) / float(w / 2))) + w / 2;
    int y = w / 2 - int(varY / (float(SCALE) / float(w / 2)));
    return  x + y * w;
}
__device__
int getBinTrace(float varTime, float varX, int w) {
    int x = int(varTime / (0.5f / float(w / 2))) + w / 2;
    int y = w / 2 - int(varX / (float(SCALE) / float(w / 2)));
    return  x + y * w;
}

__global__
void clearDensityAndOutArrKernel(uchar4* d_out, float* d_density, int w, int h, int sys) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = flatten(col, row, w, h);

    d_density[idx] = 0;
    if (sys == 0) {
        d_out[idx].x = 0;
        d_out[idx].y = 0;
        d_out[idx].z = 0;
        d_out[idx].w = 255;
    }
}

__global__
void plotKernel(uchar4* d_out, float* d_density, float3* d_particals, int w, int h, int sys) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = flatten(col, row, w, h);
    


    if (sys == 0) {// plot density
        float maximum = float(NUM);
        float l = d_density[idx] * maximum*0.2f;
        float s = 1;
        int hc = (180 + (int)(360.0f * d_density[idx] * maximum)) % 360;
        float m1, m2;

        if (l <= 5.0f)
            m2 = l * (1 + s);
        else
            m2 = l + s - l * s;
        m1 = 2 * l - m2;
        if (d_density[idx] == 0.0f) {
            d_out[idx].x = 0;
            d_out[idx].y = 0;
            d_out[idx].z = 0;
            d_out[idx].w = 255;
        }
        else {
            d_out[idx].x = value(m1, m2, hc + 120);
            d_out[idx].y = value(m1, m2, hc);
            d_out[idx].z = value(m1, m2, hc - 120);
            d_out[idx].w = 255;
        }

        if (col == h / 2 || row == w / 2) {
            d_out[idx].x = 255;
            d_out[idx].y = 255;
            d_out[idx].z = 255;
            d_out[idx].w = 255;
        }
    }
    else {// plot trace 10 particals
        if (idx < 4) {
            float x = d_particals[idx].x;
            float t = d_particals[idx].z;
            int mapIdx = getBinTrace(t, x, w);
            d_out[mapIdx].x = 255;
            d_out[mapIdx].y = 0;
            d_out[mapIdx].z = 0;
            d_out[mapIdx].w = 255;
        }
        if (col == h / 2 || row == w / 2) {
            d_out[idx].x = 255;
            d_out[idx].y = 255;
            d_out[idx].z = 255;
            d_out[idx].w = 255;
        }
    }
   
}
__global__
void makeHistKernel(float3* d_particals, float* d_density, int w, int h) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = flatten(col, row, w, h);

    float x = d_particals[idx].x, p = d_particals[idx].y;
    int mapIdx = getBin(x, p, w);
    d_density[mapIdx] += 1 / float(NUM);
}
__global__
void genRandomKernel(curandState* globalState, float3* d_particals, int w, int h) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = flatten(col, row, w, h);
    curandState localState = globalState[idx];
    //float x = curand_uniform(&localState) - 0.5f;
    //float p = curand_uniform(&localState) - 0.5f;
    float x, p = 0.0f;
    //while(idx < NUM){
    x = curand_normal(&localState);
    p = curand_normal(&localState);
    d_particals[idx].x = x;
    d_particals[idx].y = p;
    d_particals[idx].z = 0.0f;
    globalState[idx] = localState;

}
__global__
void timeNextKernel(float3* d_particals, float* d_density, int w, int h) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    const int idx = flatten(col, row, w, h);
    float x = d_particals[idx].x;
    float v = d_particals[idx].y;
    int mapIdx = getBin(x, v, w);


    float d = d_density[mapIdx], dright = d_density[mapIdx + 1], dleft = d_density[mapIdx - 1], dtop = d_density[mapIdx - w], dbottom = d_density[mapIdx + w];
    //  calculate new x and v if V == x^4

    
    float deltaX = STEPTIME * (v / MASS);
    float deltaV = STEPTIME * (-1.0f * x * x * x + x *PLANK * 1.0f / d * (dtop - 2.0f * d + dbottom) / (SCALE / (float(w) / 2.0f)));
    

    // calculate new x and v if V == Mass * omega * x^2/ 2

    //float  deltaX = STEPTIME * v;
    //float deltaV = STEPTIME * (-1.0f * x );

    d_particals[idx].x += deltaX;
    d_particals[idx].y += deltaV;
    d_particals[idx].z += STEPTIME;

}
void kernelLauncher(uchar4* d_out, float3* d_particals, float* d_density, int w, int h, int sys) {
    //for plot
    const dim3 blockSize(TX, TY);
    const dim3 gridSize(divUp(w, TX), divUp(h, TY));
    // for particles
    const dim3 blockSizePart(TX, TY);
    const dim3 gridSizePart(divUp(sqrt(NUM), TX), divUp(sqrt(NUM), TY));
    timeNextKernel <<<gridSizePart, blockSizePart >>> (d_particals, d_density, w, h);
    clearDensityAndOutArrKernel <<<gridSize, blockSize >>> (d_out, d_density, w, h, sys);
    makeHistKernel <<<gridSize, blockSize >>> (d_particals, d_density, w, h);
    plotKernel <<<gridSize, blockSize>>> (d_out, d_density, d_particals, w, h, sys);
}



void initialConditions(uchar4* d_out, float3* d_particals, float* d_density, int w, int h) {

    float3* initialRandomValues = new float3[NUM];
    FILE* infile = fopen("random.txt", "r");
    for (int i = 0; i < NUM; i++) {
        if (fscanf(infile, "%f \t %f\t %f \n", &initialRandomValues[i].x, &initialRandomValues[i].y, &initialRandomValues[i].z) == EOF) break;
    }
    fclose(infile);
    cudaMemcpy(d_particals, initialRandomValues, NUM * sizeof(float3), cudaMemcpyHostToDevice);

    const dim3 blockSize(TX, TY);
    const dim3 gridSize(divUp(w, TX), divUp(h, TY));
    makeHistKernel <<<gridSize, blockSize >>> (d_particals, d_density, w, h);
}

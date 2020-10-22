#ifndef KERNEL_H
#define KERNEL_H
#include <stdio.h>
#include <stdlib.h>
#include <random>

struct uchar4;
struct float3; // particles x and v
struct float2; // system time and number of particles
#define NUM  1000 * 1000 * 4
#define SCALE 20.0f
#define STEPTIME 0.00001f
#define MASS 0.001f
#define PLANK 1.0f

void kernelLauncher(uchar4* d_out, float3* d_particals, float* d_density, int w, int h, int sys, int* d_num);
void initialConditions(uchar4* d_out, float3* d_particals, float* d_density, int* d_num, int w, int h);
#endif

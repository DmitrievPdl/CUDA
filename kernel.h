#ifndef KERNEL_H
#define KERNEL_H
#include <stdio.h>
#include <stdlib.h>
#include <random>

struct uchar4;
struct float3; // particles x and v
struct float2; // system time and number of particles
#define NUM  256 * 16
#define SQRTNUM 16 * 4
#define SCALEY 20.0f
#define SCALEX 1.5f
#define STEPTIME 0.1f
#define MASS 2000
#define PLANK 1.0f
#define BPOT 0.2981f
#define APOT 0.2f
#define HP 0.5f
#define HX 0.05f
void kernelLauncher(uchar4* d_out, float3* d_particals, float* d_density, float* d_doubderv, float * d_dnscell, int w, int h);
void initialConditions(uchar4* d_out, float3* d_particals, float* d_density, int w, int h);
#endif


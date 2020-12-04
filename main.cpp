#include "interactions.h"
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#define ITERS_PER_RENDER 1

float x(int col) {
    return ( col * SCALE / (W / 2) - SCALE);
}
float p(int row) {
    return  (SCALE - row * SCALE / (W / 2));
}
void getParticalsForPlot(float3* particles) {
    FILE* outfile = fopen("result.txt", "w");

    for (int i = 0; i < NUM; i++) {
        if(particles[i].x != -4)
            fprintf(outfile, "%f \t %f \n", particles[i].x, particles[i].y);
    }
    fclose(outfile);
}
void findRight(float* density) {
    float dp = SCALE / (H / 2);
    float dx = SCALE / (W / 2);
    FILE* outfile = fopen("result.txt", "w");
    float f1 = 0.0f;
    float Pleft = 0.0f;
    float Pright = 0.0f;
    for (int col = 500; col < 1500; col++) {
        f1 = 0.0f;
        Pleft = 0.0f;
        Pright = 0.0f;
        
        for (int row = 500; row < 1500; row++) {
            f1 += density[row * W + col] * dp;
            Pleft += density[row * W + col - 1] * p(row) * p(row) * dp;
            Pright += density[row * W + col + 1] * p(row) * p(row) * dp;
        }
        
        fprintf(outfile, "%f \t %E \n", x(col), x(col));//1.0f / f1 * (Pleft - Pright) / (2.0f * dx));
    }
    fclose(outfile);
}

void findAveragetP(float* density) {
    float dp = SCALE / (H / 2);
    float dx = SCALE / (W / 2);
    FILE* outfile = fopen("result.txt", "w");
    float averagetV = 0.0f;
    float f1 = 0.0f;

    for (int col = 500; col < 1500; col++) {
        averagetV = 0.0f;
        f1 = 0.0f;

        for (int row = 0; row < 2000; row++) {
            averagetV += density[row * W + col] * p(row) * dp ;
        }
        for (int row = 0; row < 2000; row++) {
            f1 += density[row * W + col] * dp;
        }
        averagetV *= 1.0f / f1;
        fprintf(outfile, "%f \t %E \n", x(col), averagetV);
    }
    fclose(outfile);
}
void findTotalN(float* density) {
    float dp = SCALE / (H / 2);
    float dx = SCALE / (W / 2);
    FILE* outfile = fopen("result.txt", "at");
    float f1 = 0.0f;
    float TotalN = 0.0f;
    float dns = 0.0f;
    for (int col = 0; col < 2000; col++){
        for (int row = 0; row < 2000; row++) {
            dns += density[row * W + col] * NUM;
        }
    }
    printf("%f \n",dns);
    fprintf(outfile, "%f \n", dns);
    fclose(outfile);
}
void findTotalNgpu(float3* particles) {
    FILE* outfile = fopen("result.txt", "at");
    int totalN = 0;
    for (int i = 0; i < NUM; i++) {
        if (particles[i].z != -1)
            totalN += 1;
    }
    fprintf(outfile, "%d \n", totalN);
    fclose(outfile);
}
// texture and pixel objects
GLuint pbo = 0; // OpenGL pixel buffer object
GLuint tex = 0; // OpenGL texture object
struct cudaGraphicsResource* cuda_pbo_resource;

void render() {
    int key = 0;
    uchar4* d_out = 0;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_out, NULL, cuda_pbo_resource);

    for (int i = 0; i < ITERS_PER_RENDER; ++i) 
        kernelLauncher(d_out, d_particals, d_density, W, H, sys, d_num);
    
    if ( iterationCount == 8000) {
        float dx = SCALE / (W / 2);
        float dp = SCALE / (W / 2);
        // get particals for plot

        particles = new float3[NUM];
        cudaMemcpy(particles, d_particals, NUM * sizeof(float3), cudaMemcpyDeviceToHost);
        getParticalsForPlot(particles);
        free(particles);

        // get density for calculation
        /*
        //density = new float[W * H];
        cudaMemcpy(density, d_density, W * H * sizeof(float), cudaMemcpyDeviceToHost);

        //findRight(density);
        findAveragetV(density);
        free(density);
        */
        // find avarage V
        //density = new float[W * H];
        //cudaMemcpy(density, d_density, W * H * sizeof(float), cudaMemcpyDeviceToHost);
        //findTotalN(density);

        //particles = new float3[NUM];
        //cudaMemcpy(particles, d_particals, NUM * sizeof(float3), cudaMemcpyDeviceToHost);
        //findTotalNgpu(particles);
        //free(particles);
    }
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    char title[128];
    sprintf(title, " Iterations=%4d", iterationCount);
    glutSetWindowTitle(title);

}

void draw_texture() {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,
        GL_UNSIGNED_BYTE, NULL);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

void display() {
    render();
    draw_texture();
    glutSwapBuffers();
}

void initGLUT(int* argc, char** argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(W, H);
    glutCreateWindow("Dens. Vis.");
#ifndef __APPLE__
    glewInit();
#endif
}

void initPixelBuffer() {
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, W * H * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void exitfunc() {
    if (pbo) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
    
    cudaFree(d_density); 
    cudaFree(d_particals);
}

int main(int argc, char** argv) {
    cudaMalloc(&d_density, W * H * sizeof(float));
    cudaMalloc(&d_particals, NUM * sizeof(float3));
    cudaMalloc(&d_num, sizeof(int));
    initialConditions(d_out, d_particals, d_density, d_num, W, H);

    initGLUT(&argc, argv);
    gluOrtho2D(0, W, H, 0);
    glutKeyboardFunc(keyboard);

    glutIdleFunc(idle);

    glutDisplayFunc(display);
    initPixelBuffer();
    glutMainLoop();
    atexit(exitfunc);
    return 0;
}

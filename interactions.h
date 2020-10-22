#ifndef INTERACTIONS_H
#define INTERACTIONS_H
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define W 2000
#define H 2000
#define NUM  1000 * 1000 * 4
#define DT 1.f // source intensity increment
// if sys == 0  plot density 
// if sys == 1  plot trace
int sys = 0;

float* d_density = 0;
float* density = 0;
float3* particles = 0;
float3* d_particals = 0;
float* d_time = 0;
uchar4* d_out = 0;
int* d_num = 0;

int iterationCount = 0;

void idle(void) {
	++iterationCount;
	glutPostRedisplay();
}
void keyboard(unsigned char key, int x, int y) {
	if (key == 27) exit(0);
	if (key == '0') sys = 0;
	if (key == '1') sys = 1;
	glutPostRedisplay();
}
#endif

#include "view.h"
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

/**
Code taken from Storti and Yurtoglu's CUDA for Engineers

Use from main:
initGLUT(&argc, argv, "Title", 600, 600);
gluOrtho2D(0, 600, 600, 0);
glutDisplayFunc(display);
glutIdleFunc(update);   //your code does something in update and calls glutPostRedisplay()
initPixelBuffer();
glutMainLoop();      
atexit(exitfunc);
**/

// texture and pixel objects
GLuint pbo = 0; // OpenGL pixel buffer object
GLuint tex = 0; // OpenGL texture object
int2 loc = { 0, 0 };
unsigned int W = 0;
unsigned int H = 0;
struct cudaGraphicsResource *cuda_pbo_resource;
unsigned char *buf;

//for demo purposes
//TODO remove demo stuff
void setLoc(int x, int y) {
   loc.x = x;
   loc.y = y;
}

int getLocX() {
   return loc.x;
}

void setBuf(unsigned char *ptr) {
   buf = ptr;
}

/**
Render a new frame with kernel launch
**/
void render() {
   cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
   cudaGraphicsResourceGetMappedPointer((void **)&buf, NULL,
      cuda_pbo_resource);
   //TODO replace kernel, update args
   kernelLauncher(buf, W, H, loc);
   cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

/**
Draw rendered frame to window
**/
void drawTexture() {
   glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, W, H, 0, GL_LUMINANCE,
      GL_UNSIGNED_BYTE, 0);
   glEnable(GL_TEXTURE_2D);
   glBegin(GL_QUADS);
   glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
   glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
   glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
   glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
   glEnd();

   glDisable(GL_TEXTURE_2D);
}

/**
Render and draw a frame to the output window
**/
void display() {
   render();
   drawTexture();
   glutSwapBuffers();
}

/**
Initialize OpenGL to start using display
@param int *argc                    s
@param char **argv                  s
@param const char *title            title of display window
@param unsigned int width, height   the width and height of the display window
**/
void initGLUT(int *argc, char **argv, const char *title, unsigned int width, unsigned int height) {
   glutInit(argc, argv);
   W = width; H = height;
   glutInitDisplayMode(GLUT_LUMINANCE | GLUT_DOUBLE);
   glutInitWindowSize(W, H);
   glutCreateWindow(title);
#ifndef __APPLE__
   glewInit();
#endif

}

/**
Setup buffers for rendering frames
**/
void initPixelBuffer() {
   glGenBuffers(1, &pbo);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
   glBufferData(GL_PIXEL_UNPACK_BUFFER, (W)*H * sizeof(unsigned char), buf,
      GL_DYNAMIC_DRAW);
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_2D, tex);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
   cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
      cudaGraphicsMapFlagsWriteDiscard);
}

/**
Destroy buffers on exit
**/
void exitfunc() {
   if (pbo) {
      cudaGraphicsUnregisterResource(cuda_pbo_resource);
      glDeleteBuffers(1, &pbo);
      glDeleteTextures(1, &tex);
   }
}
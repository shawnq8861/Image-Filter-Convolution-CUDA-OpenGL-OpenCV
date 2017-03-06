#include "view.h"
#include "kernel.h"
#include "image_io.h"
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
Example driver that loads image, displays it with OpenGL, updates the images
and redisplays results. Use image /airplanes/image_0009.jpg because of padding
issues - see task list.
**/

void update() {
   setLoc(getLocX() + 1, 1);
   glutPostRedisplay();
}

void main(int argc, char **argv) {
   Mat image = loadImage("C:/Users/Alex/Pictures/CSS535Project/airplanes/image_0009.jpg"); //USE THIS IMAGE
   int cols = image.cols, rows = image.rows;
   printf("Rows %i, Cols %i\n", rows, cols);
   size_t size = rows * cols * sizeof(char);
   unsigned char *buf = new unsigned char[size];
   mat2carry(image, buf);
   setBuf(buf);

   initGLUT(&argc, argv, "Title", cols, rows);
   gluOrtho2D(0, cols, rows, 0);
   glutDisplayFunc(display);
   glutIdleFunc(update);   //your code does something in update and calls glutPostRedisplay()
   initPixelBuffer();
   glutMainLoop();
   atexit(exitfunc);
}
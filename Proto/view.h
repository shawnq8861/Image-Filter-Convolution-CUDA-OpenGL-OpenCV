#ifndef VIEW_H
#define VIEW_H

//for demo
//TODO remove demo variables and methods
void setLoc(int x, int y);
int getLocX();
void setBuf(unsigned char *ptr);

//view functions
void render();
void drawTexture();
void display();
void initGLUT(int *argc, char **argv, const char *title, unsigned int width, unsigned int height);
void initPixelBuffer();
void exitfunc();

#endif
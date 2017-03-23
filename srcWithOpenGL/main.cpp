/*
	 to compile, run make
*/

#include "kernel.h"
#include <iostream>
#include <string.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

//#include </home/ubuntu/NVIDIA_CUDA-8.0_Samples/common/inc/GL/glew.h>
#include </usr/local/cuda-8.0/samples/common/inc/GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_gl_interop.h>

using namespace std;
using namespace cv;

// texture and pixel objects
GLuint pbo = 0; // OpenGL pixel buffer object
GLuint tex = 0; // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource;

uchar *imageMatrix = NULL;
uchar *rawImage = NULL;
uchar *filteredImage = NULL;
int cols;
int rows;

void loadImageDataIntoBuffer() 
{
	unsigned char *d_imageMatrix;
	int size = rows * cols;
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_imageMatrix, NULL,
                                       cuda_pbo_resource);
	//kernelLauncher(d_out, W, H, loc);
	cudaMemcpy(d_imageMatrix, rawImage, size, cudaMemcpyHostToDevice);
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void drawImageDataAsTexture() 
{
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cols, rows, 0, GL_RGBA,
	             GL_UNSIGNED_BYTE, NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0, rows);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(cols, rows);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(cols, 0);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}

void drawTriangle()
{
    glClearColor(0.4, 0.4, 0.4, 0.4);
    glClear(GL_COLOR_BUFFER_BIT);

    glColor3f(1.0, 1.0, 1.0);
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

        glBegin(GL_TRIANGLES);
                glVertex3f(-0.7, 0.7, 0);
                glVertex3f(0.7, 0.7, 0);
                glVertex3f(0, -1, 0);
        glEnd();

    glFlush();
}

void displayImage()
{
    loadImageDataIntoBuffer();
	drawImageDataAsTexture();
	glutSwapBuffers();
}

void initPixelBuffer(int size)
{
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size, 0,
	             GL_STREAM_DRAW);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
	cudaGraphicsMapFlagsWriteDiscard);
}

void exitfunc()
{
	if (pbo) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}
}

int main(int argc, char **argv)
{
    cout << "CUDA OpenCV OpenGL Interoperability Example..." << endl;

    //uchar *imageMatrix = NULL;
	//uchar *rawImage = NULL;
	//uchar *filteredImage = NULL;

    const string imageRoot =
    "/home/ubuntu/Documents/CSS535Projects/FinalProject";

    const string imageName = "/101_ObjectCategories/airplanes/image_0014.jpg";

    const string imagePath = imageRoot + imageName;

    Mat inImgMat = imread(imagePath, IMREAD_GRAYSCALE);
	// make a copy for later...
	Mat filterImgMat(inImgMat);
    if (inImgMat.empty()) {
        cout << "error:  input image cannot be read..." << endl;
    }

    //int rows = inImgMat.rows;
    //int cols = inImgMat.cols;
	rows = inImgMat.rows;
    cols = inImgMat.cols;
	uint bitDepth = inImgMat.depth();
	int size = rows * cols;

    cout << "rows = " << rows << endl;
    cout << "cols = " << cols << endl;
	if (bitDepth == CV_8U) {
		cout << "8 bit unsigned bit depth for the image" << endl;
	}

    //namedWindow("Original Image", WINDOW_AUTOSIZE);
    //imshow("Original Image", inImgMat);

	// save original image to jpg file
	vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);
	const string origImgPath = imageRoot + "/originalImage.jpg";
    try {
    	imwrite(origImgPath, inImgMat, compression_params);
    }
	catch (Exception& ex) {
    	cout << "exception converting image to JPG format: " 
			 << ex.what() << endl;
    	return 1;
    }

	int hrzCtr = cols/2;
	int vrtCtr = rows/2;

	// transfer Mat data to CPU image buffers
	imageMatrix = new uchar[size];
	rawImage = new uchar[size];
	filteredImage = new uchar[size];
	for (int col = 0; col < rows; ++col) {
		for (int row = 0; row < cols; ++row) {
			imageMatrix[cols * col + row] = inImgMat.at<uchar>(col, row);
			rawImage[cols * col + row] = inImgMat.at<uchar>(col, row);
		}
	}

	

    // modify some values...
	
	cout << "pixel value @ (vrtCtr, hrzCtr) = " << 
			(ushort)inImgMat.at<uchar>(vrtCtr, hrzCtr) << endl;
	int radius = 30;
	for (int col = (vrtCtr - radius); col < (vrtCtr + radius); ++col) {
		for (int row = (hrzCtr - radius); row < (hrzCtr + radius); ++row) {
			imageMatrix[cols * col + row] = 255;
			//
			// two other ways of accessing the pixel value:
			//
			// *(imageMatrix + (cols * col + row)) = 255;
			// inImgMat.at<uchar>(col, row) = 255;
		}
	}

	// transfer CPU image buffer to Mat data
	for (int col = 0; col < rows; ++col) {
		for (int row = 0; row < cols; ++row) {
			inImgMat.at<uchar>(col, row) = imageMatrix[cols * col + row];
		}
	}

	cout << "pixel value @ (vrtCtr, hrzCtr) = " << 
			(ushort)inImgMat.at<uchar>(vrtCtr, hrzCtr) << endl;

	//namedWindow("Modified Image", WINDOW_AUTOSIZE);
    //imshow("Modified Image", inImgMat);

	// save modified image to jpg file
	const string modImgPath = imageRoot + "/modifiedImage.jpg";
    try {
    	imwrite(modImgPath, inImgMat, compression_params);
    }
	catch (Exception& ex) {
    	cout << "exception converting image to JPG format: " 
			 << ex.what() << endl;
    	return 1;
    }

	//
	// launch the modifyImageK() kernel function on the device (GPU)
	//
	uchar value = (uchar)0;
    modifyImage(imageMatrix, cols, rows, radius, value);

	// transfer CPU image buffer to Mat data
	for (int col = 0; col < rows; ++col) {
		for (int row = 0; row < cols; ++row) {
			inImgMat.at<uchar>(col, row) = imageMatrix[cols * col + row];
		}
	}

	cout << "pixel value @ (vrtCtr, hrzCtr) = " << 
			(ushort)inImgMat.at<uchar>(vrtCtr, hrzCtr) << endl;

	//namedWindow("Kernel Modified Image", WINDOW_AUTOSIZE);
    //imshow("Kernel Modified Image", inImgMat);

	// save kernel modified image to jpg file
	const string kModImgPath = imageRoot + "/kernelModifiedImage.jpg";
    try {
    	imwrite(kModImgPath, inImgMat, compression_params);
    }
	catch (Exception& ex) {
    	cout << "exception converting image to JPG format: " 
			 << ex.what() << endl;
    	return 1;
    }

	//
	// launch the filter kernel function on the device (GPU)
	//
	int filterKernelSize = 5;
    boxFilter(rawImage, filteredImage, cols, rows, filterKernelSize);
	
	// transfer CPU image buffer to Mat data
	for (int col = 0; col < rows; ++col) {
		for (int row = 0; row < cols; ++row) {
			filterImgMat.at<uchar>(col, row) = 
			filteredImage[cols * col + row];
		}
	}

	cout << "pixel value @ (vrtCtr, hrzCtr) = " << 
			(ushort)filterImgMat.at<uchar>(vrtCtr, hrzCtr) << endl;

	//namedWindow("Kernel Filtered Image", WINDOW_AUTOSIZE);
    //imshow("Kernel Filtered Image", inImgMat);

	// save kernel modified image to jpg file
	const string kFilterImgPath = imageRoot + "/kernelFilteredImage.jpg";
    try {
    	imwrite(kFilterImgPath, filterImgMat, compression_params);
    }
	catch (Exception& ex) {
    	cout << "exception converting image to JPG format: " 
			 << ex.what() << endl;
    	return 1;
    }

/*
	Steps to Draw an Image From CUDA

		1 Allocate a GL buffer the size of the image
		2 Allocate a GL texture the size of the image
		3 Map the GL buffer to CUDA memory
		4 Write the image from CUDA to the mapped memory
		5 Unmap the GL buffer
		6 Create the texture from the GL buffer
		7 Draw a Quad, specify the texture coordinates for each corner
		8 Swap front and back buffers to draw to the display

*/
	//
	// OpenGL interoperability example
	//
	// glut and window inititalization
	//
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE);
	glutInitWindowSize(cols, rows);
    glutInitWindowPosition(150, 300);
    glutCreateWindow("OpenGL - display image in window");
	glewInit();

	gluOrtho2D(0, cols, rows, 0);
  
  	glutDisplayFunc(drawImageDataAsTexture);
  	initPixelBuffer(size);
  	glutMainLoop();
  	atexit(exitfunc);
	


    //glutDisplayFunc(drawTriangle);
    //glutMainLoop();

	
	delete[] imageMatrix;
	delete[] rawImage;
	delete[] filteredImage;

    waitKey();

	// close and destroy the open named windows
	destroyAllWindows();

    return 0;
}

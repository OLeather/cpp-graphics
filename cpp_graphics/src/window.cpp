#include "window.hpp"
#include <iostream>

CGLWindow::CGLWindow(CGLCamera *camera, int width, int height){
    this->camera = camera;
    this->width = width;
    this->height = height;
}

int CGLWindow::init(){
    this->display = XOpenDisplay(getenv("DISPLAY"));
	if (display == NULL) {
		printf("Couldn't open display.\n");
		return -1;
	}

    int screen = DefaultScreen(display);
    this->rootWindow = DefaultRootWindow(display);
    this->window = XCreateSimpleWindow(display, rootWindow, 0, 0, width, height, 1, BlackPixel(display, screen), WhitePixel(display, screen));

    XMapWindow(display, window);
    XStoreName(display, window, "CPP Graphics");

    return 0;
}

void CGLWindow::show(){
    XSelectInput(display, window, KeyPressMask | ExposureMask);
	XEvent ev;
    this->gc = XCreateGC(display, rootWindow, 0, NULL);
	while (True) {
		int a = XNextEvent(display, &ev);
		if (ev.type == KeyPress) {
			break;
        }
		if (ev.type == Expose) {
            drawPixels(this->camera->render());
        }
    }
}

void CGLWindow::close(){
    XFreeGC(display, gc);
	XDestroyWindow(display, window);
	XCloseDisplay(display);
}

void CGLWindow::drawPixel(int x, int y, int color) {
	XSetForeground(display, gc, color);
	XDrawPoint(display, window, gc, x, y);
}

void CGLWindow::drawPixels(int*** pixels){ 
    int** pixels1 = *pixels;
    std::cout << sizeof(pixels1) / sizeof(int) << std::endl;
    
    for(int x = 0; x < camera->getWidth(); x++){
        for(int y = 0; y < camera->getHeight(); y++){
            drawPixel(x, y, (*pixels)[x][y]);
        }
    }
}
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
            auto key = XLookupKeysym(&ev.xkey, 0);
            std::cout << "Key pressed: " << key << std::endl;

            //W
            if (key == 119)  {
                this->camera->origin = this->camera->origin + Vector3(0 * cos(this->camera->rotation.x) + 1 * sin(this->camera->rotation.x), 0, 0 * sin(this->camera->rotation.x) + 1 * cos(this->camera->rotation.x));
            }
            //S
            if(key == 115){
                this->camera->origin = this->camera->origin + Vector3(0 * cos(this->camera->rotation.x) - 1 * sin(this->camera->rotation.x), 0, 0 * sin(this->camera->rotation.x) + -1 * cos(this->camera->rotation.x));    
            }
            //A
            if(key == 97){
                this->camera->origin = this->camera->origin + Vector3(-1 * cos(this->camera->rotation.x) + 0 * sin(this->camera->rotation.x), 0, 1 * sin(this->camera->rotation.x) + 0 * cos(this->camera->rotation.x));
            }
            //D
            if(key == 100){
                this->camera->origin = this->camera->origin + Vector3(1 * cos(this->camera->rotation.x) + 0 * sin(this->camera->rotation.x), 0, -1 * sin(this->camera->rotation.x) + 0 * cos(this->camera->rotation.x));
            }
            //Space
            if (key == 32)  {
                this->camera->origin = this->camera->origin + Vector3(0, 1, 0);
            }
            //Control
            if(key == 65507){
                this->camera->origin = this->camera->origin + Vector3(0, -1, 0);
            }

            //Left
            if(key == 65361){
                this->camera->rotation = this->camera->rotation + Vector3(-.1, 0, 0);
            }
            if(key == 65363){
                this->camera->rotation = this->camera->rotation + Vector3(.1, 0, 0);
            }
            if(key == 65362){

            }
            if(key == 65364){

            }
            
            drawPixels(this->camera->render());
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
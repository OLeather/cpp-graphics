#include <stdio.h>
#include <stdlib.h>
#include "X11/Xlib.h"
#include <chrono>
#include <thread>
#include <iostream>

void drawPixel(Display* display, Window window, GC gc, int x, int y, int color)
{
	XSetForeground(display, gc, color);
	XDrawPoint(display, window, gc, x, y);
}

void drawCycle(Display* display, Window window, GC gc){
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 100; j++){
            drawPixel(display, window, gc, i, j, 0x00ff00); //green
        }
    }
}

int main() 
{
	//Open Display
	Display *display = XOpenDisplay(getenv("DISPLAY"));
	if (display == NULL) {
		printf("Couldn't open display.\n");
		return -1;
	}

    int const width = 640, height = 480;
    int screen = DefaultScreen(display);
    Window rootWindow = DefaultRootWindow(display);
    Window window = XCreateSimpleWindow(display, rootWindow, 0, 0, width, height, 1, BlackPixel(display, screen), WhitePixel(display, screen));

    XMapWindow(display, window);
    XStoreName(display, window, "CPP Graphics");

    GC gc = XCreateGC(display, rootWindow, 0, NULL);

    XSelectInput(display, window, KeyPressMask | ExposureMask);
	XEvent ev;
	while (True) {
		int a = XNextEvent(display, &ev);
		if (ev.type == KeyPress) {
			break;
        }
		if (ev.type == Expose) {
            drawCycle(display, window, gc);
        }
    }

    XFreeGC(display, gc);
	XDestroyWindow(display, window);
	XCloseDisplay(display);

	return 0;
}
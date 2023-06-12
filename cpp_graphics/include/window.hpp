#ifndef CGLWINDOW_H
#define TESTCGLWINDOW_H_RENDERER_H

#include "camera.hpp"
#include "X11/Xlib.h"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

class CGLWindow {
  public:
    CGLWindow(CGLCamera *camera, int width, int height);

    int init();
    void show();
    void close();

  private:
    void drawPixels(int*** pixels);
    void drawPixel(int x, int y, int color);

    CGLCamera *camera;
    int width, height;
    Display *display;
    GC gc;
    Window window;
    Window rootWindow;
};

#endif
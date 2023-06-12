#include "renderer.hpp"
#include "X11/Xlib.h"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

class CGLWindow {
  public:
    CGLWindow(CGLRenderer *renderer, int width, int height);

    int init();
    void show();
    void close();

  private:
    void drawPixels(int*** pixels);
    void drawPixel(int x, int y, int color);

    CGLRenderer *renderer;
    int width, height;
    Display *display;
    GC gc;
    Window window;
    Window rootWindow;
};
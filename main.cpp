#include <stdio.h>
#include <stdlib.h>
#include "X11/Xlib.h"
#include <chrono>
#include <thread>
#include <iostream>
#include "cpp_graphics.hpp"

int main() {
    int width = 800, height = 800;
    float fov = 60;
    
    CGLTri *tri0 = new CGLTri(Vector3(0, 0, 3), Vector3(1, 0, 2), Vector3(1, 1, 1));

    TestRenderer *renderer = new TestRenderer(width, height);
    CGLCamera *camera = new CGLCamera(renderer, fov);
    camera->addObject(tri0);
    CGLWindow *window = new CGLWindow(camera, width, height);

    window->init();
    window->show();
    window->close();

	return 0;
}
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
    
    CGLTri *tri0 = new CGLTri(Vector3(0, 0, .3), Vector3(1, 0, .3), Vector3(1, 1, .3), DIFFUSE, 16711680, 0.8, 0.8, 5);

    // CGLTri *tri1 = new CGLTri(Vector3(.5, .5, .5), Vector3(1, 0, .5), Vector3(1, 1, .5), DIFFUSE, 16711680, 0.8, 0.8, 25);

    CGLLight *light0 = new CGLLight(Vector3(2, 0, 0), 16777215, .5);

    CGLLight *light1 = new CGLLight(Vector3(1, 1, 0), 16777215, .5);

    CGLLight *light2 = new CGLLight(Vector3(1, .5, 0), 16777215, 1);

    RayTraceRenderer *renderer = new RayTraceRenderer(width, height, 5);
    CGLCamera *camera = new CGLCamera(renderer, fov);
    camera->addObject(tri0);
    // camera->addObject(tri1);

    camera->addLight(light0);
    camera->addLight(light1);
    camera->addLight(light2);
    CGLWindow *window = new CGLWindow(camera, width, height);

    window->init();
    window->show();
    window->close();

	return 0;
}
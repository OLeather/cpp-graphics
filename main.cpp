#include <stdio.h>
#include <stdlib.h>
#include "X11/Xlib.h"
#include <chrono>
#include <thread>
#include <iostream>
#include "cpp_graphics.hpp"

int main() {
    int width = 1920, height = 1080;
    float fov = 30;
    
    // CGLTri *tri0 = new CGLTri(Vector3(10, -10, -1), Vector3(10, 0, -1), Vector3(0, 0, -1), DIFFUSE, Vector3(255, 0, 0), 0.9, 0.1, 0.001, 5);

    // CGLTri *tri1 = new CGLTri(Vector3(1.8, -1.8, -.3), Vector3(1.8, .8, -.3), Vector3(.8, .8, -.3), DIFFUSE, Vector3(0, 255, 0), 0.9, 0.1, 0.001, 5);

    float planeWidth = 50;
    float planeHeight = 50;
    Vector3 plane1Origin = Vector3(0, -5, 10);
    CGLTri *tri0 = new CGLTri(
        Vector3(-1 * planeWidth, 0, 1 * planeHeight)  + plane1Origin, 
        Vector3(-1 * planeWidth, 0, -1 * planeHeight) + plane1Origin, 
        Vector3(1 * planeWidth, 0, -1 * planeHeight)  + plane1Origin,
        DIFFUSE, 
        Vector3(255, 0, 0), 
        1, 
        0.5,
        0.2, 
        25,
        1,
        0);
        
    CGLTri *tri1 = new CGLTri(
        Vector3(1 * planeWidth, 0, 1 * planeHeight) + plane1Origin,
        Vector3(-1 * planeWidth, 0, 1 * planeHeight) + plane1Origin, 
        Vector3(1 * planeWidth, 0, -1 * planeHeight) + plane1Origin, 
        DIFFUSE, 
        Vector3(255, 255, 255), 
        0.0, 
        0.0,
        0.1, 
        25,
        .1,
        0);
    
    float plane2Width = 5;
    float plane2Height = 5;
    Vector3 plane2Origin = Vector3(0, 0, -5);
    CGLTri *tri2 = new CGLTri(
        Vector3(-1 * plane2Width, 1 * plane2Height, 0)  + plane2Origin,
        Vector3(-1 * plane2Width, -1 * plane2Height, 0) + plane2Origin, 
        Vector3(1 * plane2Width, -1 * plane2Height, 0)  + plane2Origin, 
        DIFFUSE, 
        Vector3(0, 255, 255), 
        1, 
        0.5,
        0.1, 
        25,
        1,
        0);

    // CGLTri *tri3 = new CGLTri(Vector3(-1 * plane2Width, 1 * plane2Height, 0) + plane2Origin, 
    //     Vector3(1 * plane2Width, -1 * plane2Height, 0) + plane2Origin, 
    //     Vector3(1 * plane2Width, 1 * plane2Height, 0) + plane2Origin, 
    //     DIFFUSE, 
    //     Vector3(0, 255, 0), 
    //     0.9, 
    //     0.5, 
    //     0.5, 
    //     25);
        
    CGLLight *light0 = new CGLLight(Vector3(10, 5, -10), Vector3(255, 255, 255), 1);

    CGLLight *light1 = new CGLLight(Vector3(-10, -5, -10), Vector3(255, 255, 255), 1);

    RayTraceRenderer *renderer = new RayTraceRenderer(width, height, 5);
    CGLCamera *camera = new CGLCamera(renderer, fov);

    camera->origin = Vector3(0, 0, 0);

    camera->addObject(tri0);
    camera->addObject(tri1);
    camera->addObject(tri2);
    // camera->addObject(tri3);
    // camera->addObject(tri1);

    camera->addLight(light0);
    camera->addLight(light1);

    CGLWindow *window = new CGLWindow(camera, width, height);

    window->init();
    window->show();
    window->close();

	return 0;
}
#define TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_USE_MAPBOX_EARCUT


#include <stdio.h>
#include <stdlib.h>
#include "X11/Xlib.h"
#include <chrono>
#include <thread>
#include <iostream>
#include "cpp_graphics.hpp"
#include "tiny_obj_loader.h"

int main() {
    int width = 1920, height = 1080;
    float fov = 30;
    
    // CGLTri *tri0 = new CGLTri(Vector3(10, -10, -1), Vector3(10, 0, -1), Vector3(0, 0, -1), DIFFUSE, Vector3(255, 0, 0), 0.9, 0.1, 0.001, 5);

    // CGLTri *tri1 = new CGLTri(Vector3(1.8, -1.8, -.3), Vector3(1.8, .8, -.3), Vector3(.8, .8, -.3), DIFFUSE, Vector3(0, 255, 0), 0.9, 0.1, 0.001, 5);

    std::string inputfile = "../dragon.obj";

    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile, reader_config)) {
    if (!reader.Error().empty()) {
        std::cerr << "TinyObjReader: " << reader.Error();
    }
    exit(1);
    }

    float planeWidth = 5000;
    float planeHeight = 5000;
    Vector3 plane1Origin = Vector3(0, -5, 10);
    CGLTri *tri0 = new CGLTri(
        Vector3(-1 * planeWidth, 0, 1 * planeHeight)  + plane1Origin, 
        Vector3(-1 * planeWidth, 0, -1 * planeHeight) + plane1Origin, 
        Vector3(1 * planeWidth, 0, -1 * planeHeight)  + plane1Origin,
        DIFFUSE, 
        Vector3(247, 202, 201), 
        1, 
        0.1,
        .3, 
        25,
        .1,
        .1);
        
    CGLTri *tri1 = new CGLTri(
        Vector3(1 * planeWidth, 0, 1 * planeHeight) + plane1Origin,
        Vector3(-1 * planeWidth, 0, 1 * planeHeight) + plane1Origin, 
        Vector3(1 * planeWidth, 0, -1 * planeHeight) + plane1Origin, 
        DIFFUSE, 
        Vector3(145, 168, 209), 
        1, 
        0.1,
        0.3, 
        25,
        -1,
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
        0, 
        0.5,
        0.1, 
        25,
        1.01,
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

    CGLLight *light0 = new CGLLight(Vector3(10, 1500, -10), Vector3(255, 255, 255), 1);

    CGLLight *light1 = new CGLLight(Vector3(-10, -5, 0), Vector3(255, 255, 255), 1);

    RayTraceRenderer *renderer = new RayTraceRenderer(width, height, 5);
    CGLCamera *camera = new CGLCamera(renderer, fov);

    camera->origin = Vector3(0, 600, -1500);
    camera->rotation = Vector3(.4, 0, 0);


    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    Vector3 objOrigin = Vector3(0, 0, 10);
    Vector3 objScale = Vector3(1, 1, 1);

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + 0];
                auto v1 = Vector3(attrib.vertices[3*size_t(idx.vertex_index)+0], attrib.vertices[3*size_t(idx.vertex_index)+1], attrib.vertices[3*size_t(idx.vertex_index)+2]) * objScale;
                idx = shapes[s].mesh.indices[index_offset + 1];
                auto v2 = Vector3(attrib.vertices[3*size_t(idx.vertex_index)+0], attrib.vertices[3*size_t(idx.vertex_index)+1], attrib.vertices[3*size_t(idx.vertex_index)+2]) * objScale;
                idx = shapes[s].mesh.indices[index_offset + 2];
                auto v3 = Vector3(attrib.vertices[3*size_t(idx.vertex_index)+0], attrib.vertices[3*size_t(idx.vertex_index)+1], attrib.vertices[3*size_t(idx.vertex_index)+2]) * objScale;

                 CGLTri *tri = new CGLTri(
                    v1 + objOrigin,
                    v2 + objOrigin, 
                    v3 + objOrigin, 
                    DIFFUSE,
                    Vector3(0, 255, 255), 
                    0, 
                    0.5,
                    0.2, 
                    25,
                    0,
                    0);

                camera->addObject(tri);
                index_offset += fv;
        }
    }


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
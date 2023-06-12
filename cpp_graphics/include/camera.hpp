#ifndef CGLCAMERA_H
#define CGLCAMERA_H

#include <math.h> 
#include "renderer.hpp"
#include "object.hpp"
#include <vector>

class CGLCamera {
  public:
    CGLCamera(CGLRenderer *renderer, float fov);
    int*** render();
    
    int getWidth();
    int getHeight();

    void addObject(CGLObject *object);
    void addLight(CGLLight *light);
  private:
    CGLRenderer *renderer;
    std::vector<CGLObject*> objects;
    std::vector<CGLLight*> lights;

    float fov, fx, fy, px, py;
};

#endif
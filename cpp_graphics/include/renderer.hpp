#ifndef CGLRENDERER_H
#define CGLRENDERER_H
#include "object.hpp"
#include "raytrace.cuh"
#include <vector>

class CGLRenderer {
  public:
    CGLRenderer(int width, int height);
    ~CGLRenderer();

    virtual int*** render(float fx, float fy, std::vector<CGLTri*> objects, std::vector<CGLLight*> lights) = 0;

    int getWidth();
    int getHeight();

  protected:
    int width, height;
    int** pixels;

};
#endif
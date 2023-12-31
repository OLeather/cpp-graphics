#ifndef TEST_RENDERER_H
#define TEST_RENDERER_H
#include "renderer.hpp"
#include "raytrace.cuh"

class RayTraceRenderer : public CGLRenderer {
  public:
    RayTraceRenderer(int width, int height, int steps);
    ~RayTraceRenderer();
    int*** render(float fx, float fy, Vector3 origin, Vector3 rotation, std::vector<CGLTri*> objects, std::vector<CGLLight*> lights);
  private:
    int steps;
    bool cast(const Vector3 &rayDirection, const Vector3 &rayOrigin, std::vector<CGLTri*> objects, Vector3 *hitPoint, Vector3 *hitNormal, int *hitIndex, float *depth);
    bool trace(const Vector3 &rayDirection, const Vector3 &rayOrigin, std::vector<CGLTri*> objects, std::vector<CGLLight*> lights, Vector3 *hitPoint, float *depth, int *color, int step);
    float fresnel();

};
#endif
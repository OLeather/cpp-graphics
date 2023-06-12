#ifndef TEST_RENDERER_H
#define TEST_RENDERER_H
#include "renderer.hpp"

class TestRenderer : public CGLRenderer {
  public:
    TestRenderer(int width, int height) : CGLRenderer(width, height){}
    ~TestRenderer();
    int*** render(float fx, float fy, std::vector<CGLObject*> objects);
};
#endif
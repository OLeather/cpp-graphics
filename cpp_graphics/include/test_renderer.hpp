#include "window.hpp"

class TestRenderer : public CGLRenderer {
  public:
    TestRenderer(int width, int height) : CGLRenderer(width, height){}
    ~TestRenderer();
    int*** render();
};

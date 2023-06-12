#include "test_renderer.hpp"
#include <iostream>

TestRenderer::~TestRenderer(){

}

int*** TestRenderer::render(float fx, float fy, std::vector<CGLObject*> objects){
    Vector3 *hitPoint = new Vector3();
    for(int px = 0; px < width; px++){
        for(int py = 0; py < height; py++){
            Vector3 origin = Vector3();
            float x = width/2.0 - px;
            float y = height/2.0 - py;
            float dx = x/fx;
            float dy = y/fy;
            Vector3 direction = Vector3(dx, dy, 1);
            for(auto object : objects){
                if(object->intersect(direction, origin, hitPoint)){
                    pixels[px][py] = 0x00ff00;
                }
            }
        }
    }
    return &pixels;
}

#include "test_renderer.hpp"

TestRenderer::~TestRenderer(){

}

int*** TestRenderer::render(){
    for(int x = 0; x < 100; x++){
        for(int y = 0; y < 100; y++){
            pixels[x][y] = 0x00ff00;
        }
    }
    return &pixels;
}
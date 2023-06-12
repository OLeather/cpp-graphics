#include "renderer.hpp"
#include <iostream>

CGLRenderer::CGLRenderer(int width, int height){
    this->width = width;
    this->height = height;
    this->pixels = new int*[width];
    for(int i = 0; i < width; ++i){
        this->pixels[i] = new int[height];
    }
}

CGLRenderer::~CGLRenderer(){
    
}

int CGLRenderer::getWidth(){
    return width;
}

int CGLRenderer::getHeight(){
    return height;
}


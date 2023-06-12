#include "camera.hpp"

CGLCamera::CGLCamera(CGLRenderer *renderer, float fov){
    this->renderer = renderer;
    this->fov = fov;
    fx = (renderer->getWidth()/2.0)/tan(fov/2.0);
    fy = (renderer->getWidth()/2.0)/tan(fov/2.0);
    px = renderer->getWidth()/2.0;
    py = renderer->getHeight()/2.0;
}

void CGLCamera::addObject(CGLObject *object){
    this->objects.push_back(object);
}

void CGLCamera::addLight(CGLLight *light){
    this->lights.push_back(light);
}

int*** CGLCamera::render(){
    return this->renderer->render(fx, fy, this->objects, this->lights);
}

int CGLCamera::getWidth(){
    return this->renderer->getWidth();
}

int CGLCamera::getHeight(){
    return this->renderer->getHeight();
}
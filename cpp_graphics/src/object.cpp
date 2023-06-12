#include "object.hpp"
#include <iostream>

CGLObject::CGLObject(){

}

CGLObject::~CGLObject(){

}

CGLTri::CGLTri(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2) {
    this->p0 = p0;
    this->p1 = p1;
    this->p2 = p2;

    Vector3 e0 = minus(p1, p0);
    Vector3 e1 = minus(p2, p0);
    this->n = cross(e0, e1);
}

CGLTri::~CGLTri(){

}

bool CGLTri::intersect(const Vector3 &rayDirection, const Vector3 &rayOrigin, Vector3 *hitPoint){
    //Cases where ray does not intersect: normal dot dir is close to 0 (parallel), t  < 0 (triangle behind ray), N dot (edge X v(0..2)) (passes a side)

    float nDotDir = dot(n, rayDirection);
    
    if(nDotDir < 2e-8){
        return false;
    }
    
    float t = -(dot(n, rayOrigin) - dot(n, this->p0)) / dot(n, rayDirection);
    
    if(t < 0){
        return false;
    }

    *hitPoint = plus(rayOrigin, times(rayDirection, t)); 

    if(dot(n, cross(minus(p1, p0), minus(*hitPoint, p0))) < 0 || dot(n, cross(minus(p2, p1), minus(*hitPoint, p1))) < 0 || dot(n, cross(minus(p0, p2), minus(*hitPoint, p2))) < 0){
        return false;
    }

    return true;
}
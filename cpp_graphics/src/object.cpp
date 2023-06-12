#include "object.hpp"
#include <iostream>

CGLObject::CGLObject() : type(0), color(0), diffuse(0) { }
CGLObject::CGLObject(int type) : type(type), color(0), diffuse(0) { }
CGLObject::CGLObject(int type, int color) : type(type), color(color), diffuse(0) { }
CGLObject::CGLObject(int type, int color, float diffuse, float specular, float specularExponent) : type(type), color(color), diffuse(diffuse), specular(specular), specularExponent(specularExponent) {}
CGLObject::~CGLObject() { }

int CGLObject::getType(){
    return type;
}

int CGLObject::getColor(){
    return color;
}

CGLTri::CGLTri(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, int type, int color, float diffuse, float specular, float specularExponent) : CGLObject(type, color, diffuse, specular, specularExponent) {
    this->p0 = p0;
    this->p1 = p1;
    this->p2 = p2;

    Vector3 e0 = minus(p1, p0);
    Vector3 e1 = minus(p2, p0);
    this->n = normalize(cross(e0, e1));
}

CGLTri::~CGLTri(){

}

bool CGLTri::intersect(const Vector3 &rayDirection, const Vector3 &rayOrigin, Vector3 *hitPoint, Vector3 *hitNormal, float *depth){
    //Cases where ray does not intersect: normal dot dir is close to 0 (parallel), t  < 0 (triangle behind ray), N dot (edge X v(0..2)) (passes a side)

    float nDotDir = dot(n, rayDirection);
    
    if(nDotDir < 2e-8){
        return false;
    }
    
    float t = -(dot(n, rayOrigin) - dot(n, this->p0)) / dot(n, rayDirection);
    
    if(t < 0){
        return false;
    }

    *hitPoint = plus(rayOrigin, times(t, rayDirection)); 
    *hitNormal = n;
    *depth = sqrt(pow(hitPoint->x-rayOrigin.x, 2) + pow(hitPoint->y-rayOrigin.y, 2) + pow(hitPoint->z-rayOrigin.z, 2));

    if(dot(n, cross(minus(p1, p0), minus(*hitPoint, p0))) < 0 || dot(n, cross(minus(p2, p1), minus(*hitPoint, p1))) < 0 || dot(n, cross(minus(p0, p2), minus(*hitPoint, p2))) < 0){
        return false;
    }

    return true;
}

CGLLight::CGLLight(const Vector3 &point, int color, float intensity) : CGLObject(LIGHT, color) {
    this->point = point;
    this->intensity = intensity;
}
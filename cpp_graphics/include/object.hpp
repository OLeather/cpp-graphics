#ifndef CGLOBJECT_H
#define CGLOBJECT_H
#include "vector3.hpp"
#include <math.h>

#define DIFFUSE 0
#define LIGHT 1

class CGLObject{
  public:
    CGLObject();
    CGLObject(int type);
    CGLObject(int type, Vector3 color);
    CGLObject(int type, Vector3 color, float diffuse, float specular, float ambient, float shinyness);
    ~CGLObject();

    virtual bool intersect(const Vector3 &rayDirection, const Vector3 &rayOrigin, Vector3 *hitPoint, Vector3 *hitNormal, float *depth) = 0;

    int getType();

    int type;
    Vector3 color;
    float diffuse, specular, ambient, shinyness;
};

class CGLTri : public CGLObject{
  public:
    CGLTri(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, int type, Vector3 color, float diffuse, float specular, float ambient, float shinyness); 
    ~CGLTri();
    bool intersect(const Vector3 &rayDirection, const Vector3 &rayOrigin, Vector3 *hitPoint, Vector3 *hitNormal, float *depth);
   
    Vector3 p0, p1, p2, n;
};

class CGLLight : public CGLObject{
  public:
    CGLLight(const Vector3 &point, Vector3 color, float intensity);
    bool intersect(const Vector3 &rayDirection, const Vector3 &rayOrigin, Vector3 *hitPoint, Vector3 *hitNormal, float *depth) {return false;}
    Vector3 point;
    float intensity;
};

#endif
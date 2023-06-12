#ifndef CGLOBJECT_H
#define CGLOBJECT_H
#include "vector3.hpp"

class CGLObject{
  public:
    CGLObject();
    ~CGLObject();

    virtual bool intersect(const Vector3 &rayDirection, const Vector3 &rayOrigin, Vector3 *hitPoint) = 0;
};

class CGLTri : public CGLObject{
  public:
    CGLTri(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2); 
    ~CGLTri();
    bool intersect(const Vector3 &rayDirection, const Vector3 &rayOrigin, Vector3 *hitPoint);
   
    Vector3 p0, p1, p2, n;
};

#endif
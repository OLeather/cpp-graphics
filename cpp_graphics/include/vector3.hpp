#ifndef VECTOR3_H
#define VECTOR3_H
class Vector3{
  public:
    Vector3();
    Vector3(float x, float y, float z);
    float x, y, z;
};


float dot(const Vector3 &v0, const Vector3 &v1);

Vector3 cross(const Vector3 &v0, const Vector3 &v1);

Vector3 plus(const Vector3 &v0, const Vector3 &v1);

Vector3 minus(const Vector3 &v0, const Vector3 &v1);

Vector3 times(const Vector3 &v0, const float s);

#endif
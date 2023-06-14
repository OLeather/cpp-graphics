#ifndef VECTOR3_H
#define VECTOR3_H
class Vector3{
  public:
    Vector3();
    Vector3(float x, float y, float z);
    float x, y, z, magnitude;

    Vector3 operator+(const Vector3 &v);
    Vector3 operator-(const Vector3 &v);
    Vector3 operator*(const Vector3 &v);
    Vector3 operator*(const float s);

    void asFloatArray(float *arr) const;
};

float dot(const Vector3 &v0, const Vector3 &v1);

Vector3 cross(const Vector3 &v0, const Vector3 &v1);

Vector3 plus(const Vector3 &v0, const Vector3 &v1);

Vector3 minus(const Vector3 &v0, const Vector3 &v1);

Vector3 times(const float s, const Vector3 &v0);

Vector3 times(const Vector3 &v0, const Vector3 &v1);

Vector3 normalize(const Vector3 &v);

Vector3 reflect(const Vector3 &dir, const Vector3 &normal);

#endif
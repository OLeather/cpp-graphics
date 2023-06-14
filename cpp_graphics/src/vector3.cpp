#include "vector3.hpp"
#include <math.h>

Vector3::Vector3() : x(0), y(0), z(0), magnitude(0) {}
Vector3::Vector3(float x, float y, float z) : x(x), y(y), z(z), magnitude(sqrt(x*x+y*y+z*z)) {}

Vector3 Vector3::operator+(const Vector3 &v){
    return plus(*this, v);
}

Vector3 Vector3::operator-(const Vector3 &v){
    return minus(*this, v);
}

Vector3 Vector3::operator*(const Vector3 &v){
    return times(*this, v);
}

Vector3 Vector3::operator*(const float s){
    return times(s, *this);
}

void Vector3::asFloatArray(float *arr) const{
    arr[0] = x;
    arr[1] = y;
    arr[2] = z;
}

float dot(const Vector3 &v0, const Vector3 &v1) {
    return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z;
}

Vector3 cross(const Vector3 &v0, const Vector3 &v1){
    return Vector3(v0.y*v1.z-v0.z*v1.y, -(v0.x*v1.z-v0.z*v1.x), v0.x*v1.y-v0.y*v1.x);
}

Vector3 plus(const Vector3 &v0, const Vector3 &v1){
    return Vector3(v0.x+v1.x, v0.y+v1.y, v0.z+v1.z);
}

Vector3 minus(const Vector3 &v0, const Vector3 &v1){
    return Vector3(v0.x-v1.x, v0.y-v1.y, v0.z-v1.z);
}

Vector3 times(const float s, const Vector3 &v0){
    return Vector3(v0.x*s, v0.y*s, v0.z*s);
}

Vector3 times(const Vector3 &v0, const Vector3 &v1){
    return Vector3(v0.x*v1.x, v0.y*v1.y, v0.z*v1.z);
}

Vector3 normalize(const Vector3 &v){
    return times(1.0/v.magnitude, v);
}

Vector3 reflect(const Vector3 &dir, const Vector3 &normal){
    // return minus(dir, times(2.0, times(dot(dir, normalize(normal)), normalize(normal))));
    return minus(times(2.0, times(dot(dir, normal), normal)), dir);
}
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector3.hpp>
#include <vector>
#include "object.hpp"


namespace Raytrace {
	struct Object;
	struct Triangle;
	struct Sphere;
	struct Light;

	void render(int width, int height, float fx, float fy, Vector3 origin, Vector3 rotation, std::vector<CGLTri*> objects, std::vector<CGLLight*> lights, int** pixels);
	__global__ void renderKernel(float fx, float fy, int width, int height, float3 origin, float3 rotation, Triangle tris[], int numTris, Sphere spheres[], int numSpheres, Light lights[], int numLights, int* pixels);
	__device__ void raytrace(float3 dir, float3 origin,  Triangle tris[], int numTris, Sphere spheres[], int numSpheres, Light lights[], int numLights, int &color, int step);
	__device__ float cast(float3 dir, float3 origin, Triangle tris[], int numTris, Sphere spheres[], int numSpheres);
	__device__ float cast(float3 dir, float3 origin, Triangle tris[], int numTris, Sphere spheres[], int numSpheres, float3 &hitPoint, float3 &hitNormal, int &hitIndex);

	const int MAX_RAYTRACE_DEPTH = 4;
}

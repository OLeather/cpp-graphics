#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector3.hpp>
#include <vector>
#include "object.hpp"


namespace Raytrace {
	struct Triangle;
	struct Light;

	void render(int width, int height, float fx, float fy, std::vector<CGLTri*> objects, std::vector<CGLLight*> lights, int** pixels);
	__global__ void trace(float fx, float fy, int width, int height, Triangle tris[], int numTris, Light lights[], int numLights, int* pixels);
	__device__ void raytrace(float3 dir, float3 origin, Triangle tris[], int numTris, Light lights[], int numLights, int &color, int step);
	__device__ void cast(float3 dir, float3 origin, Triangle tris[], int numTris, float3 &hitPoint, float3 &hitNormal, int &hitIndex, float &depth);
}

#include "raytrace.cuh"
#include <iostream>

#include <stdio.h>

inline __host__ __device__ float3 vec_to_float3(Vector3 v)
{
    return make_float3(v.x, v.y, v.z);
}

inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __host__ __device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __host__ __device__ float3 operator*(const float3 &a, const float3 &b) {
  return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __host__ __device__ float3 operator*(const float3 &a, const float &s) {
  return make_float3(a.x*s, a.y*s, a.z*s);
}

inline __host__ __device__ float3 operator*(const float &s, const float3 &a) {
  return make_float3(a.x*s, a.y*s, a.z*s);
}

inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

inline __host__ __device__ float magnitude(float3 v)
{
    return sqrt(dot(v, v));
}

inline __host__ __device__ float3 reflect(float3 dir, float3 normal){
	return (dot(normalize(dir),  normalize(normal)) * 2.0 * normalize(normal)) - normalize(dir);
}

namespace Raytrace {
	struct Triangle {
		float3 v0, v1, v2, n, color;
		float diffuse, specular, ambient, shinyness, reflectivity, transparency;
		Triangle() {}
		Triangle(float3 v0, float3 v1, float3 v2, float3 color, float diffuse, float specular, float ambient, float shinyness, float reflectivity, float transparency) : v0(v0), v1(v1), v2(v2), n(normalize(cross(v1-v0, v2-v0))), color(color), diffuse(diffuse), specular(specular), ambient(ambient), shinyness(shinyness), reflectivity(reflectivity), transparency(transparency) {}

		__device__ float intersect(float3 dir, float3 origin, float3 &hitPoint, float3 &hitNormal){
			//Cases where ray does not intersect: normal dot dir is close to 0 (parallel), t  < 0 (triangle behind ray), N dot (edge X v(0..2)) (passes a side)

			float nDotDir = dot(n, dir);
			
			if(abs(nDotDir) < 2e-8){
				return -1;
			}
			
			float t = -(dot(n, origin) - dot(n, v0)) / dot(n, dir);
			
			if(t < 0){
				return -1;
			}

			hitPoint = origin + dir * t; 
			hitNormal = n;
			float depth = magnitude(hitPoint-origin);

			if(dot(n, cross(v1-v0, hitPoint-v0)) < 0 || dot(n, cross(v2-v1, hitPoint-v1)) < 0 || dot(n, cross(v0-v2, hitPoint- v2)) < 0){
				return -1;
			}

			return depth;
		}
	};
	

	struct Light{
		float3 point, color;
		float intensity;
		Light(){}
		Light(float3 point, float3 color, float intensity) : point(point), color(color), intensity(intensity) {}
	};

	void render(int width, int height, float fx, float fy, Vector3 origin, Vector3 rotation, std::vector<CGLTri*> objects, std::vector<CGLLight*> lights, int** pixels){
		int h_numTris = objects.size();
		Triangle h_tris[h_numTris];
		for(int i = 0; i < objects.size(); i++){
			Triangle tri = Triangle(vec_to_float3(objects[i]->p0), vec_to_float3(objects[i]->p1), vec_to_float3(objects[i]->p2), vec_to_float3(objects[i]->color), objects[i]->diffuse, objects[i]->specular, objects[i]->ambient, objects[i]->shinyness, objects[i]->reflectivity, objects[i]->transparency);
			h_tris[i] = tri;
		}

		int h_numLights = lights.size();
		Light h_lights[h_numLights];
		for(int i = 0; i < lights.size(); i++){
			Light light = Light(vec_to_float3(lights[i]->point), vec_to_float3(lights[i]->color), lights[i]->intensity);
			h_lights[i] = light;
		}

		Triangle *d_tris;
		Light *d_lights;
		int *d_pixels;
		int *h_pixels = (int*)malloc(width*height*sizeof(int));
		
		cudaMalloc(&d_tris, sizeof(h_tris));
		cudaMalloc(&d_lights, sizeof(h_lights));
		cudaMalloc(&d_pixels, width*height*sizeof(int));
		
		cudaMemcpy(d_tris, h_tris, sizeof(h_tris), cudaMemcpyHostToDevice);
		cudaMemcpy(d_lights, h_lights, sizeof(h_lights), cudaMemcpyHostToDevice);

		// render each pixel
		int N = width*height;
    	int thr_per_blk = 256;
    	int blk_in_grid = ceil( float(N) / thr_per_blk );

		float3 d_origin = make_float3(origin.x, origin.y, origin.z);
		float3 d_rotation = make_float3(rotation.x, rotation.y, rotation.z);

		// std::cout << thr_per_blk << std::endl;

		trace<<<blk_in_grid, thr_per_blk>>>(fx, fy, width, height, d_origin, d_rotation, d_tris, h_numTris, d_lights, h_numLights, d_pixels);

		cudaMemcpy(h_pixels, d_pixels, width*height*sizeof(int), cudaMemcpyDeviceToHost);

		for(int id = 0; id < width*height; id++){
			int px = id % width;
			int py = id / width;

			pixels[px][py] = h_pixels[id];

			// std::cout << px << " " << py << " " <<  h_pixels[id] << std::endl;
		}
		
		cudaFree(d_tris);
	    cudaFree(d_lights);
	}

	__device__ unsigned long createRGB(int r, int g, int b)
	{   
		return ((r & 0xff) << 16) + ((g & 0xff) << 8) + (b & 0xff);
	}

	__global__ void trace(float fx, float fy, int width, int height, float3 origin, float3 rotation, Triangle tris[], int numTris, Light lights[], int numLights, int* pixels)
	{
		int id = blockDim.x * blockIdx.x + threadIdx.x;
		int px = id % width;
		int py = id / width;

		float x =  px - width/2.0;
		float y =  height/2.0 - py;
		float dx = x/-fx;
		float dy = y/-fy;
		float3 dir = normalize(make_float3(dx * cos(rotation.x) + 1 * sin(rotation.x), dy, -dx * sin(rotation.x) + 1 * cos(rotation.x)));
		

		int color = 0;
		raytrace(dir, origin, tris, numTris, lights, numLights, color, 0);
		
		pixels[id] = color;
	}

	__device__ void raytrace(float3 dir, float3 origin, Triangle tris[], int numTris, Light lights[], int numLights, int &color, int step){
		float3 hitPoint;
		float3 N;
		int hitIndex;
		float depth = 0;

		color = createRGB(0, 0, 0);

		cast(dir, origin, tris, numTris, hitPoint, N, hitIndex, depth);
		if(depth > 0 && depth > 0.01){
			if(step > 2){
				color = createRGB(min(int(tris[hitIndex].color.x), 255), min(int(tris[hitIndex].color.y), 255), min(int(tris[hitIndex].color.z), 255));
			}
			else{
				float3 V = normalize(origin-hitPoint);

				float3 diffuse = make_float3(0, 0, 0);
				float3 specular = make_float3(0, 0, 0);

				float3 hitOffset = (dot(dir, N) < 0) ? (hitPoint + N * 0.001) : (hitPoint - N * 0.001);
				for(int i = 0; i < numLights; i++){
					float3 L = normalize(lights[i].point - hitPoint);

					float3 R = reflect(-L, N);
					float3 shadowPoint;
					float3 shadowNormal;
					int shadowIndex;
					float shadowDepth;

					float lightDistance = magnitude(lights[i].point - hitPoint);

					cast(L, hitOffset, tris, numTris, shadowPoint, shadowNormal, shadowIndex, shadowDepth);
					bool shadow = shadowDepth != -1 && shadowDepth < lightDistance;
					if(!shadow){
						diffuse += tris[hitIndex].color * lights[i].intensity *  max(0.0f, dot(N, -L));
						specular += lights[i].color * lights[i].intensity * pow(max(0.0f,dot(R, dir)), tris[hitIndex].shinyness);
					}
				}

				float3 W = reflect(V, N);
				int reflectColor = 0;
				raytrace(W, hitOffset, tris, numTris, lights, numLights, reflectColor, step + 1);
				float r = ((reflectColor >> 16) & 0xff);
				float g = ((reflectColor >>  8) & 0xff);
				float b = ((reflectColor      ) & 0xff);
				float3 reflectedLight = tris[hitIndex].reflectivity * make_float3(r, g, b);

				float3 directLight = tris[hitIndex].color * tris[hitIndex].ambient + (diffuse * tris[hitIndex].diffuse) + (specular * tris[hitIndex].specular);

				float3 colorVec = directLight + reflectedLight;

				color = createRGB(min(int(colorVec.x), 255), min(int(colorVec.y), 255), min(int(colorVec.z), 255));
			}
		}
	}

	__device__ void cast(float3 dir, float3 origin, Triangle tris[], int numTris, float3 &hitPoint, float3 &hitNormal, int &hitIndex, float &depth){
		float closestDepth = 9999999;
		float3 point;
		float3 normal;
		depth = -1;
		
		for(int i = 0; i < numTris; i++){
			float depth_ = tris[i].intersect(dir, origin, point, normal);
			if(depth_ > 0 && depth_ < closestDepth){
				closestDepth = depth_;
				depth = depth_;
				hitPoint = point;
				hitIndex = i;
				hitNormal = normal;
			} 
		}
	}
}
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

inline __host__ __device__ float3 min(float a, float3 v)
{
    return make_float3(min(a, v.x), min(a, v.y), min(a, v.z));
}

inline __host__ __device__ float3 max(float a, float3 v)
{
    return make_float3(max(a, v.x), max(a, v.y), max(a, v.z));
}

inline __host__ __device__ float3 reflect(float3 dir, float3 normal){
	return (dot(normalize(dir),  normalize(normal)) * 2.0 * normalize(normal)) - normalize(dir);
}

namespace Raytrace {
	struct Triangle {
		float3 v0, v1, v2, n, color;
		float diffuse, specular, ambient, shinyness, ior, transparency;
		Triangle() {}
		Triangle(float3 v0, float3 v1, float3 v2, float3 color, float diffuse, float specular, float ambient, float shinyness, float ior, float transparency) : v0(v0), v1(v1), v2(v2), n(normalize(cross(v1-v0, v2-v0))), color(color), diffuse(diffuse), specular(specular), ambient(ambient), shinyness(shinyness), ior(ior), transparency(transparency) {}

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

	struct Sphere{
		float3 center, color;
		float radius, diffuse, specular, ambient, shinyness, ior, transparency;
		Sphere() {}
		Sphere(float3 center, float radius, float3 color, float diffuse, float specular, float ambient, float shinyness, float ior, float transparency) : center(center), radius(radius), color(color), diffuse(diffuse), specular(specular), ambient(ambient), shinyness(shinyness), ior(ior), transparency(transparency) {}

		__device__ int solveQuadratic(float &a, float &b, float &c, float &x0, float &x1){
			float d = b * b - 4 * a * c;
			
			if(d < 0) {
				return -1;
			}
			else if(d == 0){
				x0 = x1 = -0.5 * b/a;
			}
			else{
				float q = (b > 0) ? 
					-0.5 * (b - sqrt(d)) :
					-0.5 * (b - sqrt(d));
				x0 = q/a;
				x1 = c/q;
			}
			if(x0 > x1){
				float temp = x1;
				x1 = x0;
				x0 = temp;
			}

			return 1;
		}

		__device__ float intersect(float3 dir, float3 origin, float3 &hitPoint, float3 &hitNormal){
			float3 L = origin-center;
			float a = dot(dir, dir);
			float b = 2 * dot(dir, L);
			float c = dot(L, L) - (radius*radius);
			float t0 = 0;
			float t1 = 0;
			if(solveQuadratic(a, b, c, t0, t1) == -1){
				return -1;
			}
			if(t0 > t1){
				float temp = t1;
				t1 = t0;
				t0 = temp;
			}

			if(t0 < 0){
				t0 = t1;
				if(t0 < 0){
					return -1;
				}
			}

			hitPoint = origin + dir * t0; 
			hitNormal = normalize(hitPoint-center);

			return magnitude(hitPoint-origin);
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
			Triangle tri = Triangle(vec_to_float3(objects[i]->p0), vec_to_float3(objects[i]->p1), vec_to_float3(objects[i]->p2), vec_to_float3(objects[i]->color), objects[i]->diffuse, objects[i]->specular, objects[i]->ambient, objects[i]->shinyness, objects[i]->ior, objects[i]->transparency);
			h_tris[i] = tri;
		}

		int h_numSpheres = 5;
		Sphere h_spheres[h_numSpheres];
		for(int i = 0; i < h_numSpheres; i++){
			Sphere sphere = Sphere(make_float3(i * 15, 0, 5), 5, make_float3(255, 255, 255), 0, 1, 0.0, 50, .1, 0);
			h_spheres[i] = sphere;
		}

		int h_numLights = lights.size();
		Light h_lights[h_numLights];
		for(int i = 0; i < lights.size(); i++){
			Light light = Light(vec_to_float3(lights[i]->point), vec_to_float3(lights[i]->color), lights[i]->intensity);
			h_lights[i] = light;
		}

		Triangle *d_tris;
		Sphere *d_spheres;
		Light *d_lights;
		int *d_pixels;
		int *h_pixels = (int*)malloc(width*height*sizeof(int));
		
		cudaMalloc(&d_tris, sizeof(h_tris));
		cudaMalloc(&d_spheres, sizeof(h_spheres));
		cudaMalloc(&d_lights, sizeof(h_lights));
		cudaMalloc(&d_pixels, width*height*sizeof(int));
		
		cudaMemcpy(d_tris, h_tris, sizeof(h_tris), cudaMemcpyHostToDevice);
		cudaMemcpy(d_spheres, h_spheres, sizeof(h_spheres), cudaMemcpyHostToDevice);
		cudaMemcpy(d_lights, h_lights, sizeof(h_lights), cudaMemcpyHostToDevice);

		// render each pixel
		int N = width*height;
    	int thr_per_blk = 256;
    	int blk_in_grid = ceil( float(N) / thr_per_blk );

		float3 d_origin = make_float3(origin.x, origin.y, origin.z);
		float3 d_rotation = make_float3(rotation.x, rotation.y, rotation.z);

		// std::cout << thr_per_blk << std::endl;

		renderKernel<<<blk_in_grid, thr_per_blk>>>(fx, fy, width, height, d_origin, d_rotation, d_tris, h_numTris, d_spheres, h_numSpheres, d_lights, h_numLights, d_pixels);

		cudaMemcpy(h_pixels, d_pixels, width*height*sizeof(int), cudaMemcpyDeviceToHost);

		for(int id = 0; id < width*height; id++){
			int px = id % width;
			int py = id / width;

			pixels[px][py] = h_pixels[id];

			// std::cout << px << " " << py << " " <<  h_pixels[id] << std::endl;
		}
		
		cudaFree(d_tris);
		cudaFree(d_spheres);
	    cudaFree(d_lights);
	}

	__device__ unsigned long createRGBInt(int r, int g, int b) {   
		return ((r & 0xff) << 16) + ((g & 0xff) << 8) + (b & 0xff);
	}

	__device__ float3 createRGBVec(int color) {
		return make_float3(((color >> 16) & 0xff), ((color >>  8) & 0xff), ((color) & 0xff));
	}

	__device__ float fresnel(const float3 &dir, const float3 &N, const float &ior){
		float kr = 0;
		float cosi = dot(dir, N);
		if(cosi > 1) cosi = 1;
		if(cosi < -1) cosi = -1;

		float etai = 1, etat = ior;

		if(cosi > 0){
			float temp = etai;
			etai = etat;
			etat = temp;
		}

		float sint = etai / etat * sqrt(max(0.0f, 1-cosi*cosi));

		if(sint >= 1){
			kr = 1;
		}
		else{
			float cost = sqrt(max(0.0f, 1-sint*sint));
			cosi = abs(cosi);
			float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
			float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
			kr = (Rs * Rs + Rp * Rp) / 2;
		}

		return kr;
	}

	__global__ void renderKernel(float fx, float fy, int width, int height, float3 origin, float3 rotation, Triangle tris[], int numTris, Sphere spheres[], int numSpheres, Light lights[], int numLights, int* pixels)
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
		raytrace(dir, origin, tris, numTris, spheres, numSpheres, lights, numLights, color, 0);
		
		pixels[id] = color;
	}

	__device__ void raytrace(float3 dir, float3 origin, Triangle tris[], int numTris, Sphere spheres[], int numSpheres, Light lights[], int numLights, int &color, int step){
		float3 hitPoint;
		float3 N;
		int hitIndex;
		float depth = 0;

		color = createRGBInt(0, 0, 0);

		if(step < MAX_RAYTRACE_DEPTH){
			depth = cast(dir, origin, tris, numTris, spheres, numSpheres, hitPoint, N, hitIndex);
			if(depth > 0 && depth > 0.01){
				float3 hitColor;
				float kDiffuse, kSpecular, kAmbient, shinyness, kIndexOfRefraction;
				if(hitIndex < numTris){
					hitColor = tris[hitIndex].color;
					kDiffuse = tris[hitIndex].diffuse;
					kSpecular = tris[hitIndex].specular;
					kAmbient = tris[hitIndex].ambient;
					shinyness = tris[hitIndex].shinyness;
					kIndexOfRefraction = tris[hitIndex].ior;
				}
				else{
					int index = hitIndex-numTris;
					hitColor = spheres[index].color;
					kDiffuse = spheres[index].diffuse;
					kSpecular = spheres[index].specular;
					kAmbient = spheres[index].ambient;
					shinyness = spheres[index].shinyness;
					kIndexOfRefraction = spheres[index].ior;
				}

				float3 V = normalize(origin-hitPoint);

				float3 diffuseVec = make_float3(0, 0, 0);
				float3 specularVec = make_float3(0, 0, 0);

				float3 hitOffset = (dot(dir, N) < 0) ? (hitPoint + N * 0.001) : (hitPoint - N * 0.001);
				for(int i = 0; i < 1; i++){
					float3 L = normalize(lights[i].point - hitPoint);

					float RdotDir = max(0.0f, dot(reflect(-L, N), dir));
					float N_dot_L = max(0.0f, dot(N, -L));

					float lightDistance = magnitude(lights[i].point - hitPoint);
					float shadowDepth = cast(L, hitOffset, tris, numTris, spheres, numSpheres);
					
					bool shadow = shadowDepth != -1 && shadowDepth < lightDistance;
	
					if(!shadow){
						diffuseVec += hitColor * lights[i].intensity *  N_dot_L;
						specularVec += lights[i].color * lights[i].intensity * pow(RdotDir, shinyness);
					}
				}

				// printf("%f %f %f \n", diffuseVec.x, diffuseVec.y, diffuseVec.z);

				float3 W = reflect(V, N);
				int reflectColor = 0;
				
				raytrace(W, hitOffset, tris, numTris, spheres, numSpheres, lights, numLights, reflectColor, step + 1);
				float kr = fresnel(dir, N, kIndexOfRefraction);
				float3 reflectedLight = kr * createRGBVec(reflectColor);

				float3 directLight = hitColor * kAmbient + (diffuseVec * kDiffuse) + (specularVec * kSpecular);

				float3 colorVec = directLight + reflectedLight;

				color = createRGBInt(min(int(colorVec.x), 255), min(int(colorVec.y), 255), min(int(colorVec.z), 255));
			}
		}	
	}

	__device__ float cast(float3 dir, float3 origin, Triangle tris[], int numTris, Sphere spheres[], int numSpheres){
		float closestDepth = -1;
		float3 point;
		float3 normal;
		
		for(int i = 0; i < numTris; i++){
			float depth = tris[i].intersect(dir, origin, point, normal);
			if(depth > 0 && (depth < closestDepth || closestDepth == -1)){
				closestDepth = depth;
			} 
		}
		for(int i = 0; i < numSpheres; i++){
			float depth = spheres[i].intersect(dir, origin, point, normal);
			if(depth > 0 && (depth < closestDepth || closestDepth == -1)){
				closestDepth = depth;
			} 
		}

		return closestDepth;
	}

	__device__ float cast(float3 dir, float3 origin, Triangle tris[], int numTris, Sphere spheres[], int numSpheres, float3 &hitPoint, float3 &hitNormal, int &hitIndex){
		float closestDepth = -1;
		float3 point;
		float3 normal;
		
		for(int i = 0; i < numTris; i++){
			float depth = tris[i].intersect(dir, origin, point, normal);
			if(depth > 0 && (depth < closestDepth || closestDepth == -1)){
				closestDepth = depth;
				hitPoint = point;
				hitIndex = i;
				hitNormal = normal;
			} 
		}
		for(int i = 0; i < numSpheres; i++){
			float depth = spheres[i].intersect(dir, origin, point, normal);
			if(depth > 0 && (depth < closestDepth || closestDepth == -1)){
				closestDepth = depth;
				hitPoint = point;
				hitIndex = i + numTris;
				hitNormal = normal;
			} 
		}

		return closestDepth;
	}
}
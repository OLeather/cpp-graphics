# Recursive Ray Tracing implemented in C++

![gif](https://github.com/OLeather/cpp-graphics/blob/main/ray_trace_gif_high_quality.gif)

To expand upon a previous project I made a few years ago in High School, [OLeather/java-3d-renderer](https://github.com/OLeather/java-3d-renderer), I decided to use my broader set of software experience now as a University student to re-approach the project in C++. I also used it as an exercise to learn CUDA, which is widely used in the field of machine learning. My new approach was to implement a recursive ray tracer, which can achieve much more realistic lighting effects, as well as support for reflective and transparent materials using ray reflection and refraction.

## Features
Capable of diffuse, specular, reflective, and refractive light calculations for a variety of effective materials and surfaces. Computes each pixel in parallel using CUDA graphics acceleration.

![Screenshot 2023-06-18 002757](https://github.com/OLeather/cpp-graphics/assets/43189206/53fefe1d-3f24-461e-8c80-b483a321afb5)

## Algorithms
The diffuse and specular light are calculated using the Phong illumination model. For reflection, the ray from the pixel to the object is reflected across the object's normal, and the ray tracing algorithm is run recursively on the new reflected ray. For refraction, the refractive ray is calculated based on the hit normal and the index of refraction of the object the ray hits, and the ray tracing algorithm is again run recursively on the new refracted ray. The ray is traced recursively up to 4 times. The reflected and refracted ray intensity is calculated using the fresnel effect for more realistic lighting.

The implementation of the renderer can be found in [cpp-renderer/src/raytrace.cu](https://github.com/OLeather/cpp-graphics/blob/main/cpp_graphics/src/raytrace.cu)

Parellel computation of rays:
```c++
// Number of threads = number of pixels to compute
int N = width*height;
// Use arbitrary 256 threads per block
int thr_per_blk = 256;
// Calculate blocks in the compute grid
int blk_in_grid = ceil( float(N) / thr_per_blk );

// Call the render kernel on the GPU to compute each pixel
renderKernel<<<blk_in_grid, thr_per_blk>>>(fx, fy, width, height, d_origin, d_rotation, d_tris, h_numTris, d_spheres, h_numSpheres, d_lights, h_numLights, d_pixels);
cudaDeviceSynchronize(); 
```

```c++
__global__ void renderKernel( ... ) {
  // Get thread index
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  // Get pixel from thread
  int px = i % width;
  int py = i / width;

  // Create ray from pixel
  float x =  px - width/2.0;
  float y =  height/2.0 - py;
  float dx = x/-fx;
  float dy = y/-fy;
  float dz = 1;
  
  ...

  // Trace ray and return final pixel color
  raytrace(dir, origin, tris, numTris, spheres, numSpheres, lights, numLights, color, 0);
}
```

Direct illumination:
```c++
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

float3 directLight = hitColor * kAmbient + (diffuseVec * kDiffuse) + (specularVec * kSpecular);
```

Reflection:
```c++
int reflectColor = 0;

float kr = fresnel(dir, N, kIndexOfRefraction);
float3 reflectionDirection = reflect(V, N);
raytrace(reflectionDirection, hitOffset, tris, numTris, spheres, numSpheres, lights, numLights, reflectColor, step + 1);
float3 reflectedLight = kr * createRGBVec(reflectColor);
```

Refraction:
```c++
int refractColor = 0;

float kr = fresnel(dir, N, kIndexOfRefraction);
float3 refractionDirection = normalize(refract(dir, N, kIndexOfRefraction));
float3 refractionOrigin = outside ? hitPoint - bias : hitPoint + bias;
raytrace(refractionDirection, refractionOrigin, tris, numTris, spheres, numSpheres, lights, numLights, refractColor, step + 1);
float3 refractedLight = (1.0-kr) * createRGBVec(refractColor);
```

final color:
```c++
float3 color = directLight + reflectedLight + refractedLight;
```

# Results
![Screenshot 2023-06-16 151325](https://github.com/OLeather/cpp-graphics/assets/43189206/c3b09c4f-2770-47c1-8396-91482b4ee9fa)
![Screenshot 2023-06-16 125026](https://github.com/OLeather/cpp-graphics/assets/43189206/3b135cdf-4411-4359-9f3b-18662d61aa98)

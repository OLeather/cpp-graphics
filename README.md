# Recursive Ray Tracing implemented in C++

![gif](https://github.com/OLeather/cpp-graphics/blob/main/ray_trace_gif_high_quality.gif)

## Features
Capable of diffuse, specular, reflective, and refractive light calculations for a variety of effective materials and surfaces. Computes each pixel in parallel using CUDA graphics acceleration.

## Algorithms
The diffuse and specular light are calculated using the Phong illumination model. For reflection, the ray from the pixel to the object is reflected across the object's normal, and the ray tracing algorithm is run recursively on the new reflected ray. For refraction, the refractive ray is calculated based on the hit normal and the index of refraction of the object the ray hits, and the ray tracing algorithm is again run recursively on the new refracted ray. The ray is traced recursively up to 4 times.

![Screenshot 2023-06-16 151325](https://github.com/OLeather/cpp-graphics/assets/43189206/c3b09c4f-2770-47c1-8396-91482b4ee9fa)
![Screenshot 2023-06-16 125026](https://github.com/OLeather/cpp-graphics/assets/43189206/3b135cdf-4411-4359-9f3b-18662d61aa98)

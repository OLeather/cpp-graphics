#include "raytrace_renderer.hpp"
#include <iostream>
#include <limits>
#include <memory>
#include <cuda.h>

RayTraceRenderer::RayTraceRenderer(int width, int height, int steps) : CGLRenderer(width, height) {
    this->steps = steps;
}

RayTraceRenderer::~RayTraceRenderer(){

}

bool RayTraceRenderer::cast(const Vector3 &rayDirection, const Vector3 &rayOrigin, std::vector<CGLTri*> objects, Vector3 *hitPoint, Vector3 *hitNormal, int *hitIndex, float *depth){
    float closestDepth = std::numeric_limits<float>::infinity();
    bool hit = false;
    Vector3 *tempPt = new Vector3();
    Vector3 *tempNormal = new Vector3();
    for(int i = 0; i < objects.size(); i++){
        if(objects[i]->intersect(rayDirection, rayOrigin, tempPt, tempNormal, depth)){
            if(*depth < closestDepth){
                closestDepth = *depth;
                *hitIndex = i;
                *hitPoint = *tempPt;
                *hitNormal = *tempNormal;
            }
            hit = true;
        }
    }
    return hit;
}

unsigned long createRGB(int r, int g, int b)
{   
    return ((r & 0xff) << 16) + ((g & 0xff) << 8) + (b & 0xff);
}

bool RayTraceRenderer::trace(const Vector3 &rayDirection, const Vector3 &rayOrigin, std::vector<CGLTri*> objects, std::vector<CGLLight*> lights, Vector3 *hitPoint, float *depth, int *color, int step){
    int *hitIndex = new int;
    Vector3 *hitNormal = new Vector3();
    if(cast(rayDirection, rayOrigin, objects, hitPoint, hitNormal, hitIndex, depth)){
        Vector3 illumination = Vector3();
        Vector3 hitOffset = (dot(rayDirection, *hitNormal) < 0) ? (*hitPoint + times(0.000001, *hitNormal)) : (*hitPoint - times(0.000001, *hitNormal));
        Vector3 ambientLight = Vector3();
        for(auto light : lights){
            Vector3 lightDirection = light->point - *hitPoint; 
            float lightDistance = lightDirection.magnitude;
            lightDirection = normalize(lightDirection);
            float LdotN = std::max(0.0f, -dot(lightDirection, *hitNormal));
            
            // std::cout << lightDirection.x << "  " << lightDirection.y << " " << lightDirection.z  << " " <<  hitPoint->x << "  " << hitPoint->y << " " << hitPoint->z << std::endl;
        
            Vector3 *shadowHitPoint = new Vector3();
            Vector3 *shadowHitNormal = new Vector3();
            int *shadowHitIndex = new int;
            float *shadowHitDepth = new float;
            bool shadow = cast(lightDirection, hitOffset, objects, shadowHitPoint, shadowHitNormal, shadowHitIndex, shadowHitDepth) &&  *shadowHitDepth < lightDistance;
            // bool shadow = false;
            // std::cout << cast(lightDirection, hitOffset, objects, shadowHitPoint, shadowHitNormal, shadowHitIndex, shadowHitDepth) << std::endl;
            if(!shadow){
                // illumination = plus(illumination, times(objects[*hitIndex]->diffuse, times(light->intensity, times(LdotN, light->color))));
                illumination = illumination + ((light->color * light->intensity) * LdotN) * objects[*hitIndex]->diffuse;
            }
            else{
                // std::cout << "shadow" << std::endl;
            }
            
            // ambientLight = ambientLight + (light->color * light->intensity);
            // Vector3 reflectionDirection = reflect(Vector3() - lightDirection, *hitNormal);

            // float RmDotV = std::max(0.0f, -dot(reflectionDirection, rayDirection));
            // illumination = illumination + objects[*hitIndex]->color * pow(RmDotV, objects[*hitIndex]->shinyness) * objects[*hitIndex]->specular;
            // specularColor += pow(std::max(0.0f, -dot(reflectionDirection, rayDirection)), objects[*hitIndex]->shinyness) * light->intensity;
        }

        // *color = lightAmt * objects[*hitIndex]->color * objects[*hitIndex]->diffuse + specularColor * objects[*hitIndex]->specular;
        // Vector3 vecColor = plus(times((times(objects[*hitIndex]->ambient, ambientLight)), objects[*hitIndex]->color),illumination);
        Vector3 vecColor = (objects[*hitIndex]->color) * objects[*hitIndex]->ambient + illumination;
        *color = createRGB(std::min(255, int(vecColor.x)), std::min(255, int(vecColor.y)), std::min(255, int(vecColor.z)));
        // *color = createRGB(255, 255, 255);
        return true;
    }

    return false;
}

int*** RayTraceRenderer::render(float fx, float fy, std::vector<CGLTri*> objects, std::vector<CGLLight*> lights){
    
    Vector3 *hitPoint = new Vector3();
    float *depth = new float;
    int *color = new int;
    // for(int px = 0; px < width; px++){
    //     for(int py = 0; py < height; py++){
    //         Vector3 origin = Vector3();
    //         float x =  px - width/2.0;
    //         float y =  height/2.0 - py;
    //         float dx = x/-fx;
    //         float dy = y/-fy;
    //         Vector3 direction = Vector3(dx, dy, 1);
    //         // Raytrace::trace(direction, origin, objects, lights, color);
    //         // std::cout << *color << std::endl;
    //         // if(trace(direction, origin, objects, lights, hitPoint, depth, color, 0)){
    //         //     pixels[px][py] = *color;
    //         // }
    //         if(*color > 0){
    //             pixels[px][py] = *color;
    //         }
    //     }
    //     std::cout << px << " " << std::endl;
    // }

    Raytrace::render(width, height, fx, fy, objects, lights, pixels);

    std::cout << "Done" << std::endl;
    return &pixels;
}

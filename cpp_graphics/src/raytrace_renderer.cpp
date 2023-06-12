#include "raytrace_renderer.hpp"
#include <iostream>
#include <limits>
#include <memory>

RayTraceRenderer::RayTraceRenderer(int width, int height, int steps) : CGLRenderer(width, height) {
    this->steps = steps;
}

RayTraceRenderer::~RayTraceRenderer(){

}

bool RayTraceRenderer::cast(const Vector3 &rayDirection, const Vector3 &rayOrigin, std::vector<CGLObject*> objects, Vector3 *hitPoint, Vector3 *hitNormal, int *hitIndex, float *depth){
    float closestDepth = std::numeric_limits<float>::infinity();
    bool hit = false;
    for(int i = 0; i < objects.size(); i++){
        if(objects[i]->intersect(rayDirection, rayOrigin, hitPoint, hitNormal, depth)){
            if(*depth < closestDepth){
                closestDepth = *depth;
                *hitIndex = i;
                hit = true;
            }
        }
    }
    return hit;
}

bool RayTraceRenderer::trace(const Vector3 &rayDirection, const Vector3 &rayOrigin, std::vector<CGLObject*> objects, std::vector<CGLLight*> lights, Vector3 *hitPoint, float *depth, int *color, int step){
    int *hitIndex = new int;
    Vector3 *hitNormal = new Vector3();
    if(cast(rayDirection, rayOrigin, objects, hitPoint, hitNormal, hitIndex, depth)){
        float lightAmt = 0, specularColor = 0;
        Vector3 hitOffset = (dot(rayDirection, *hitNormal) < 0) ? (times(0.000001, minus(*hitPoint,*hitNormal))) : (times(0.000001, plus(*hitPoint,*hitNormal)));
        
        for(auto light : lights){
            Vector3 lightDirection = minus(light->point, *hitPoint); 
            float lightDistanceSquared = dot(lightDirection, *hitNormal);
            lightDirection = normalize(lightDirection);
            float LdotN = std::max(0.0f, dot(lightDirection, *hitNormal));

            Vector3 *shadowHitPoint = new Vector3();
            Vector3 *shadowHitNormal = new Vector3();
            int *shadowHitIndex = new int;
            float *shadowHitDepth = new float;
            bool shadow = cast(lightDirection, hitOffset, objects, shadowHitPoint, shadowHitNormal, shadowHitIndex, shadowHitDepth) && (*shadowHitDepth)*(*shadowHitDepth) < lightDistanceSquared;
            if(!shadow){
                lightAmt += light->intensity * light->getColor() * LdotN;
            }
            Vector3 reflectionDirection = reflect(minus(Vector3(),lightDirection), *hitNormal);
            specularColor += pow(std::max(0.0f, -dot(reflectionDirection, rayDirection)), objects[*hitIndex]->specularExponent) * light->intensity;
        }

        *color = lightAmt * objects[*hitIndex]->color * objects[*hitIndex]->diffuse + specularColor * objects[*hitIndex]->specular;

        return true;
    }

    return false;
}

int*** RayTraceRenderer::render(float fx, float fy, std::vector<CGLObject*> objects, std::vector<CGLLight*> lights){
    Vector3 *hitPoint = new Vector3();
    float *depth = new float;
    int *color = new int;
    for(int px = 0; px < width; px++){
        for(int py = 0; py < height; py++){
            Vector3 origin = Vector3();
            float x = width/2.0 - px;
            float y = height/2.0 - py;
            float dx = x/fx;
            float dy = y/fy;
            Vector3 direction = Vector3(dx, dy, 1);
            if(trace(direction, origin, objects, lights, hitPoint, depth, color, 0)){
                pixels[px][py] = *color;
            }
        }
    }
    return &pixels;
}

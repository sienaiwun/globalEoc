
#include <optix.h>
#include <optixu/optixu_math_namespace.h>



using namespace optix;

rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, reflectors, , );
rtDeclareVariable(uint, max_depth, , );


rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

__device__ float  myfmax(float a, float b) {
	return ((a) > (b) ? a : b);
}
struct PerRayData_shadow
{
	float3 attenuation;
	float3 worldPos;
	float t_hit;
};
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(float3, lightPos, , );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, diffuse_Color, , );
RT_PROGRAM void closest_hit_radiance()
{
	float3 hit_point = ray.origin + t_hit * ray.direction;

	float3 L = normalize(lightPos - hit_point);
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float kd = myfmax(dot(world_shade_normal, L), 0);
	float ka = 0.2;
	float3 color = (kd + ka)*make_float3(diffuse_Color.x, diffuse_Color.y, diffuse_Color.z);

	prd_shadow.worldPos = hit_point;
	prd_shadow.attenuation = color;;
}


#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;
__device__ float  myfmax(float a, float b) {
	return ((a) > (b) ? a : b);
}

struct PerRayData_shadow
{
	float3 attenuation;
	float t_hit;
};


rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

rtTextureSampler<float4, 2> diffuse_texture;
rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, lightPos, , );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(int3, index_color, attribute index_color, );

rtDeclareVariable(uint, max_depth, , );
rtDeclareVariable(uint, radiance_ray_type, , );
rtDeclareVariable(uint, shadow_ray_type, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, reflectors, , );

RT_PROGRAM void closest_hit_radiance()
{
	prd_shadow.t_hit = t_hit;
	float3 hit_point = ray.origin + t_hit * ray.direction;
	float3 L = normalize(lightPos - hit_point);
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float kd = myfmax(dot(world_shade_normal, L), 0);
	float ka = 0.2;
	float3 color = make_float3((kd + ka)*tex2D(diffuse_texture, texcoord.x, texcoord.y));
	prd_shadow.attenuation = color;
}

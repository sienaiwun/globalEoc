
/*
* Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and proprietary
* rights in and to this software, related documentation and any modifications thereto.
* Any use, reproduction, disclosure or distribution of this software and related
* documentation without an express license agreement from NVIDIA Corporation is strictly
* prohibited.
*
* TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
* SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
* LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
* BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
* INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGES
*/

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include <optixu/optixu_matrix_namespace.h>

using namespace optix;

rtTextureSampler<float4, 2>  request_texture;

rtBuffer<float4, 2>          result_buffer;
rtDeclareVariable(uint, shadow_ray_type, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(float3, light_pos, , );
rtDeclareVariable(rtObject, reflectors, , );
rtDeclareVariable(float3, eye_pos, , );
rtDeclareVariable(float3, eoc_eye_pos, , );
rtDeclareVariable(optix::Matrix4x4, optixModeView_Inv, , );
rtDeclareVariable(float2, resolution, , );
struct PerRayData_shadow
{
	float3 attenuation;
	float t_hit;
};


rtDeclareVariable(float2, bbmin, , );
rtDeclareVariable(float2, bbmax, , );
__device__ float3 getImagePos(float2 tc)
{
	float2 xy = bbmin + (bbmax - bbmin)*tc;
	xy = xy;
	float4 temp = make_float4(xy.x, xy.y, -1, 1)*optixModeView_Inv;
	temp = temp / temp.w;
	return make_float3(temp.x, temp.y, temp.z);
}


RT_PROGRAM void shadow_request()
{
	float2 tc = make_float2(launch_index.x, launch_index.y) / resolution;
	float4 textValue = tex2D(request_texture, launch_index.x, launch_index.y);
	if (textValue.x >= 1.0)
	{
		float3 targetPos = make_float3(textValue.y, textValue.z, textValue.w);
		float3 ray_origin = eoc_eye_pos;
		PerRayData_shadow prd;
		prd.attenuation = make_float3(-1);
		float3 L = targetPos - ray_origin;
		float dist = sqrtf(dot(L, L));
		float3 ray_direction = L / dist;
		optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, shadow_ray_type, scene_epsilon, textValue.x);
		rtTrace(reflectors, ray, prd);
		result_buffer[launch_index] = make_float4(prd.attenuation,1);
		result_buffer[launch_index].z = 1;
		return;
	}
	result_buffer[launch_index] = textValue;
}



RT_PROGRAM void exception()
{
}

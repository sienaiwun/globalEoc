#version 430
layout(early_fragment_tests) in;
in vec3 worldPos;
in vec3 worldNormal;
in vec2 tc;
in int out_vertexId;

layout( location = 0 ) out vec4 FragColor0;




layout (binding = 0, r32ui) uniform uimage2D head_pointer_image;
layout (binding = 1, rgba32ui) uniform uimageBuffer list_buffer;
//layout (binding = 2, rgba32f)  writeonly uniform coherent image2D out_texture;
//layout (binding = 0, offset = 0) uniform atomic_uint index_counter;
layout (binding = 2, r32ui) uniform uimage2D atomic_counter_array_buffer_texture;

uniform sampler2D objectTex;
uniform vec3 objectDiffuseColor;
uniform mat4 modelView; // Projection * ModelView

uniform vec3 cameraPos;
uniform vec3 lightPos;
uniform int hasTex;
uniform int objectId;
uniform float reflectFactor;


void main()
{
	float reflectValue = reflectFactor;
	vec3 N = normalize(worldNormal);
	vec3 L = normalize(lightPos-worldPos);
	float kd=max(dot(N,L),0);
	vec3 diffuse;
	if(bool(hasTex))
	{
		diffuse = texture2D(objectTex,tc).xyz;
	}
	else
	{
		diffuse = objectDiffuseColor;
	}
	float ka = 0.2;
	//FragColor0.xyz = diffuse*(kd+ka)+vec3(0.2,0.2,0.2);
	vec4 fragColor;
	fragColor.xyz = diffuse*(kd+ka)+vec3(0.2,0.2,0.2);
	fragColor.w = 1;
	//imageStore(out_texture, ivec2(gl_FragCoord.xy), fragColor);

	uvec4 item;
	item.x = packUnorm4x8(vec4(N,1));
	item.y = packUnorm4x8(fragColor);
	item.z = floatBitsToUint(gl_FragCoord.z / gl_FragCoord.w);
	item.w = 0;


	uint index;
	uint offset;

		
	//uint index = atomicCounterIncrement(index_counter);
    //uint old_head = imageAtomicExchange(head_pointer_image, ivec2(gl_FragCoord.xy), index);
	
	index = imageAtomicAdd(atomic_counter_array_buffer_texture, ivec2(gl_FragCoord.xy), 1);
	offset = (uint(gl_FragCoord.y) * 800 + uint(gl_FragCoord.x))*7;
	//offset = imageLoad(head_pointer_image, ivec2(gl_FragCoord.xy));
	imageStore(list_buffer, int(index+offset), item);
	
	
	


}

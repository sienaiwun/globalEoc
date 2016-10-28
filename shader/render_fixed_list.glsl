#version 430

#define MAX_FRAGMENTS_LAYERS 16

layout(local_size_x = 16, local_size_y = 16) in;


layout (binding = 0, r32ui) uniform uimage2D head_pointer_image;
layout (binding = 1, rgba32ui) uniform uimageBuffer list_buffer;
layout (binding = 2, r32ui) uniform uimage2D atomic_counter_array_buffer_texture;
layout (binding = 3, rgba32f) writeonly uniform image2D out_texture;

uvec4 fragments[MAX_FRAGMENTS_LAYERS];

void build_local_fragments_list(uint offset, uint frag_count) {
	uint current;

	uint i;
	for(i = 0; i < frag_count && i < MAX_FRAGMENTS_LAYERS; i++) {
		uvec4 item = imageLoad(list_buffer, int(offset+i));
		fragments[i] = item;
	}

}


void sort_fragments_list(uint frag_count) {
	uint i,j;
	uvec4 tmp;

	// INSERTION SORT
	for(i = 1; i < frag_count; ++i) {
		tmp = fragments[i];
		for(j = i; j > 0 && tmp.z > fragments[j-1].z; --j) {
			fragments[j] = fragments[j-1];
		}
		fragments[j] = tmp;
	}
}

vec4 blend(vec4 current_color, vec4 new_color) {
	return mix(current_color, new_color, new_color.a);
}

vec4 calculate_final_color(uint offset, uint frag_count) {
	
	vec4 final_color = vec4(0);
	for(uint i = 0; i < frag_count; i++) {
		uvec4 item = fragments[i];
		vec4 frag_color = unpackUnorm4x8(item.y);
		frag_color.w  = 0.2;
		final_color = blend(final_color, frag_color);
	}
	return final_color;

}

void main()
{

	uint offset;// = imageLoad(head_pointer_image, ivec2(gl_GlobalInvocationID.xy));
	offset = (uint(gl_GlobalInvocationID.y) * 800 + uint(gl_GlobalInvocationID.x))*7;
	uint frag_count = imageLoad(atomic_counter_array_buffer_texture, ivec2(gl_GlobalInvocationID.xy)).x;

	build_local_fragments_list(offset, frag_count);
	sort_fragments_list(frag_count);
	vec4 diffuse = calculate_final_color(offset,frag_count);



//	int frag_count = build_local_fragments_list();
//	sort_fragments_list(frag_count);
//	vec4 diffuse = calculate_final_color(frag_count);
//	vec4 diffuse = vec4(frag_count/16.0,0,0,1);
//	vec2 xy = gl_GlobalInvocationID.xy;
	imageStore(out_texture, ivec2(gl_GlobalInvocationID.xy), diffuse);
}
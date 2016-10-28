"#version 430                       \
#define MAX_FRAGMENTS 16             \
layout(local_size_x = 1, local_size_y = 16) in;            \
layout (binding = 0, rgba32f) writeonly uniform image2D out_texture;\
void main()\
{\
	vec4 pixel = vec4(1.0, 1.0,1.0, 1.0);\
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);\
	imageStore(out_texture, pixel_coords, pixel);\
}\n"
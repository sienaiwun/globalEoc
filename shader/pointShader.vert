#version 420

layout (location = 0) in vec3  position;
uniform mat4 MVP; // Projection * ModelView
uniform sampler2D colorTex;
uniform sampler2D worldPosTex;
out vec2 tc;

void main()
{

    tc = position.xy;
	vec3 temp = texture2D(worldPosTex,tc).xyz;
	if(length(temp)<1)
	{
		gl_Position =  vec4(-2,-3,-2,1);
		return;
	}
	gl_Position =  MVP * vec4(temp,1);
	
	
}
#version 430

layout(location = 0) in vec2 tc;

uniform sampler2D posTex;

uniform sampler2D progTex;

layout (location =  0) out vec4 color0;  

void main()
{
		vec4 posValue = texture2D(posTex,tc);
		vec4 progValue = texture2D(progTex,tc);
		if(progValue.x>0.5)
		{
			color0.xyz =progValue.yzw;
		}
		else
		{
			color0.xyz = posValue.xyz;
		}
		color0.w = 1;
}
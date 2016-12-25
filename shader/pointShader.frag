#version 430

layout(location = 0) in vec2 tc;

uniform sampler2D colorTex;
uniform sampler2D worldPosTex;

layout (location =  0) out vec4 color0;  

void main()
{
		color0.xyz = texture2D(colorTex,tc).xyz;
		color0.w = 1;
}
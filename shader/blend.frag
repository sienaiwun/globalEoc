#version 430

layout(location = 0) in vec2 tc;

uniform sampler2D color1Tex;

uniform sampler2D color2Tex;

layout (location =  0) out vec4 color0;  

void main()
{
		color0.xyz = texture2D(color1Tex,tc).xyz*0.5;
		color0.xyz += texture2D(color2Tex,tc).xyz*0.5;
		color0.w = 1;
		
}
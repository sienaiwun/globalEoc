#version 430

layout(location = 0) in vec2 tc;

uniform sampler2D colorTex;


layout (location =  0) out vec4 color0;  

void main()
{
		color0 = texture2D(colorTex,tc);
		
		
}
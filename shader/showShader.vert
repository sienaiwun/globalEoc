#version 420



uniform vec2 begin;
uniform vec2 end;


layout (location = 0) in vec2  position;


layout (location = 0 ) out vec2 tc;

void main()
{

	
	vec2 ndc = begin +(end-begin)*( position /2.0 + 0.5);
	gl_Position = vec4(ndc*2-vec2(1,1),0,1.0);
	
	tc = (position /2.0) + 0.5;
	
}
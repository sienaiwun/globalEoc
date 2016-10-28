#version 420


layout (location = 0) in vec2  position;


layout (location = 0 ) out vec2 tc;

void main()
{

	gl_Position = vec4(position,0,1.0);
	
	tc = (position /2.0) + 0.5;

}
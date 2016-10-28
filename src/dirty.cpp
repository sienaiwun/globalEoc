#include <GL/glew.h>
#include <stdlib.h>
#include <stdio.h>
#include<GL/wglew.h>
void externWglewInit()
{
	wglSwapIntervalEXT(0);
}
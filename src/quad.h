#include "scene.h"
#include "Camera.h"
#ifndef QUAD_H
#define QUAD_H

class QuadScene :public Scene
{
public:
	QuadScene()
	{
		m_fileName.resize(1);
		m_fileName[0] = "./model/urban/quad.obj";
		init();
	};
};
#endif
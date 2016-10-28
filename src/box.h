#include "scene.h"
#include "Camera.h"
#ifndef BOX_H
#define BOX_H

class boxScene :public Scene
{
public:
	boxScene()
	{
		m_fileName.resize(1);
		m_fileName[0] = "./model/urban/sphere.obj";
		m_cameraFile = "./model/urban/box.txt";
		init();
	};
};
#endif
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
		m_fileName[0] = "./model/urban/box.obj";
		m_cameraFile = "./model/urban/box.txt";
		//m_naviFile = "./model/urban/boxnaviOneWayOk.txt";
		m_naviFile = "./model/urban/boxnaviTopHalfRight.txt";
		m_ray_type = primary_ray_e;
		init();
	};
};
#endif 
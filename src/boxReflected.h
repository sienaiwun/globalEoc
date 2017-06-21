#include "scene.h"
#include "Camera.h"
#ifndef BOXREFLECTED_H
#define BOXREFLECTED_H

class boxReflectedScene :public Scene
{
public:
	boxReflectedScene()
	{
		m_fileName.resize(1);
		m_fileName[0] = "./model/urban/reflection_plane.obj";
		m_cameraFile = "./model/urban/box.txt";
		//m_naviFile = "./model/urban/boxnaviOneWayOk.txt";
		m_naviFile = "./model/urban/boxnaviRightTop.txt";
		m_ray_type = reflected_ray_e;
		init();
	};
};
#endif 
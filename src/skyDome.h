#include "scene.h"
#include "Camera.h"
#ifndef SKY_DOME_H
#define SKY_DOME_H

class SkeyDomeScene :public Scene
{
public:
	SkeyDomeScene()
	{
		m_fileName.resize(1);
		m_fileName[0] = "./model/skyDome/skyDome.obj";
		m_cameraFile = "./model/urban/box.txt";
		m_naviFile = "./model/urban/boxnavi.txt";
		init();
	};
};
#endif
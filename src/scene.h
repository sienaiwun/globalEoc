#include "myGeometry.h"
#include "glslShader.h"
#include "textureManager.h"
#include <string>
#include <vector>
#include "Camera.h"
#include "macro.h"
#ifdef OPTIX
	#include <optixu/optixpp_namespace.h>
	#include <optixu/optixu_math_namespace.h>
	using namespace optix;
	#include <cuda.h>
	#include <cuda_runtime.h>

#endif


#ifndef SCENE_H
#define SCENE_H
class Scene
{
public:
	Scene() :m_objNum(0)
	{
		m_cameraFile = std::string("");
		m_naviFile = std::string("");
	}
	virtual void init();
	virtual void render(glslShader & shader, textureManager& manager, Camera * pCamera);
	inline nv::vec3f getLightPos()
	{
		return m_lightPos;
	}
	void LoadCamera(Camera * pCamera)
	{
		if (m_cameraFile.compare("") == 0)
		{

		}
		else
		{
			pCamera->loadToFIle(m_cameraFile.c_str());
		}
	}
	void LoadNaviCam(Camera* pCamera)
	{
		if (m_naviFile.compare("") == 0)
		{

		}
		else
		{
			pCamera->loadToFIle(m_naviFile.c_str());
		}
	}
	std::string getCameraFile() const
	{
		return m_cameraFile;
	}
	std::string getNaviCamerFile() const
	{
		return m_naviFile;
	}
	Ray_type getRayType()
	{
		return m_ray_type;
	}

	
protected:
	int m_objNum;
	vector<myGeometry*> m_geometryVec;
	vector<string> m_fileName;
	nv::vec3f m_lightPos;
	std::string m_cameraFile;
	std::string m_naviFile;
	Ray_type m_ray_type ;
private:
#ifdef OPTIX
public:
	inline void setOptix(optix::Context * p)
	{
		pContext = p;
	}
	virtual void optixInit();
private:
	optix::GeometryGroup m_geometrygroup;
	optix::Context *pContext;
#endif
};
#endif
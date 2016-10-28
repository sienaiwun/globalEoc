#include "scene.h"
#include <algorithm>

#define CHECK_ERRORS()         \
	do {                         \
	GLenum err = glGetError(); \
	if (err) {                                                       \
	printf( "GL Error %d at line %d of FILE %s\n", (int)err, __LINE__,__FILE__);       \
	exit(-1);                                                      \
		}                                                                \
		} while(0)

void Scene::render(glslShader & shader, textureManager& manager, Camera * pCamera)
{
	//auto func = std::bind(&myGeometry::render, std::placeholders::_1, 222);
	//std::for_each(m_geometryVec.begin(), m_geometryVec.end(), func);
	shader.begin();
	CHECK_ERRORS();
	shader.setScene(this);
	CHECK_ERRORS();
	shader.setCamera(pCamera);
	CHECK_ERRORS();
	auto function = [&shader, &manager](myGeometry* pGeometry){pGeometry->render(shader, manager);};
	CHECK_ERRORS();
	std::for_each(m_geometryVec.begin(), m_geometryVec.end(), function);
	CHECK_ERRORS();
	shader.end();
	CHECK_ERRORS();
}
void Scene::init()
{
	m_lightPos = nv::vec3f(20, 20, 20);
	m_geometryVec.resize(m_fileName.size());
	for (int i = 0; i < m_geometryVec.size(); i++)
	{
		m_geometryVec[i] = new myGeometry(m_fileName[i].c_str());
		m_geometryVec[i]->Create_GPU_Buffer();
		m_objNum += m_geometryVec[i]->getSize();
	}
}
#ifdef OPTIX
void Scene::optixInit()
{
	std::vector<optix::GeometryInstance> geoInstance;

	geoInstance.resize(m_objNum);
	int index = 0;
	for (int i = 0; i<m_geometryVec.size(); i++)
	{
		for (int modelNum = 0; modelNum < m_geometryVec[i]->getSize(); modelNum++)
		{
			geoInstance[index++] = m_geometryVec[i]->getInstance(modelNum);
		}
	}
	m_geometrygroup = (*pContext)->createGeometryGroup();
	m_geometrygroup->setChildCount(m_objNum);
	index = 0;
	for (int i = 0; i<m_objNum; i++)
		m_geometrygroup->setChild(i, geoInstance[i]);
	m_geometrygroup->setAcceleration((*pContext)->createAcceleration("Bvh", "Bvh"));
	(*pContext)["reflectors"]->set(m_geometrygroup);
	(*pContext)["lightPos"]->set3fv((const float*)&m_lightPos);
	
}
#endif
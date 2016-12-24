#include "Constructor.h"
void Constructor::init()
{
	m_construct_Fbo.init();
	m_constuctResource = new CudaPboResource();
	m_constuctResource->set(m_width, m_height, construct_t);
	m_constuctResource->init();
	m_getTex = m_constuctResource->getTexture();
	construct_cudaInit();
}
void Constructor::render(glslShader & shader, textureManager& manager)
{
	m_construct_Fbo.begin();
	m_pScene->render(shader, manager, m_pNaviCam);   
	//nv::vec4f point = m_construct_Fbo.debugPixel(1, 412, 512);
	//m_construct_Fbo.SaveBMP("save.bmp",0);
	m_construct_Fbo.end();
}
void Constructor::construct()
{
	m_constuctResource->map();
	mapConstruct(m_pNaviCam);
	cuda_Construct(m_width,m_height);
	m_constuctResource->unMap();
	CHECK_ERRORS();
	m_constuctResource->generateTex();
	CHECK_ERRORS();
}
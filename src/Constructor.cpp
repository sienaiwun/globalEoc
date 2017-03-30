#include "Constructor.h"
void Constructor::init()
{
	m_construct_Fbo.init();
	m_constuctResource = new CudaPboResource();
	m_constuctResource->set(m_width, m_height, construct_t);
	m_constuctResource->init();
	m_getTex = m_constuctResource->getTexture();
	construct_cudaInit();


	m_optixColorResource = new  CudaTexResourse();
	m_optixColorResource->set(m_optixColorTex, m_optixWidth, m_optixHeight, optixColorTex_t);
	m_optixColorResource->init();

	m_posBlendTex = new CudaTexResourse();
	m_posBlendTex->set(pPosBlendFbo->getTexture(0), m_width, m_height, posBlend_t);
	m_posBlendTex->init();

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
	m_optixColorResource->map();
	m_posBlendTex->map();
	const int eocWidhth = ROWLARGER*m_width;
	const int eocHeight = ROWLARGER*m_height;
	edgeRendering(eocWidhth, eocHeight);
	cuda_Construct(m_width,m_height);

	m_posBlendTex->unmap();
	m_optixColorResource->unmap();
	m_constuctResource->unMap();
	CHECK_ERRORS();

	m_constuctResource->generateTex();
	CHECK_ERRORS();
}
#include "Constructor.h"

extern void constructInit(Ray_type ray_type);
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


	m_gbufferGeoResource = new CudaTexResourse();
	m_gbufferGeoResource->set(m_gbufferGeoTex, m_naviWidth, m_naviHeight, g_buffer_pos_t);
	m_gbufferGeoResource->init();

	m_gbufferNorResource = new CudaTexResourse();
	m_gbufferNorResource->set(m_gbufferNorTex, m_naviWidth, m_naviHeight, g_buffer_nor_t);
	m_gbufferNorResource->init();
	constructInit(m_pScene->getRayType());

}

void Constructor::optixInit()
{

	glGenBuffers(1, &m_resultPbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_resultPbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * sizeof(float4), 0, GL_STREAM_READ);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glGenTextures(1, &m_resultTex);
	glBindTexture(GL_TEXTURE_2D, m_resultTex);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);


	try
	{
		
		(*m_pOptixContex)["reflected_ray_type"]->setUint(1u);
		(*m_pOptixContex)["construct_res"]->setFloat(m_width, m_height);
		m_rtTexture = (*m_pOptixContex)->createTextureSamplerFromGLImage(m_construct_Fbo.getTexture(1), RT_TARGET_GL_TEXTURE_2D);
		m_rtTexture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
		m_rtTexture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
		m_rtTexture->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
		m_rtTexture->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
		m_rtTexture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
		m_rtTexture->setMaxAnisotropy(1.0f);
		m_rtTexture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
		(*m_pOptixContex)["pos_texture"]->setTextureSampler(m_rtTexture);


		m_normalTexture = (*m_pOptixContex)->createTextureSamplerFromGLImage(m_construct_Fbo.getTexture(2), RT_TARGET_GL_TEXTURE_2D);
		m_normalTexture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
		m_normalTexture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
		m_normalTexture->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
		m_normalTexture->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
		m_normalTexture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
		m_normalTexture->setMaxAnisotropy(1.0f);
		m_normalTexture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
		(*m_pOptixContex)["normal_texture"]->setTextureSampler(m_normalTexture);

		
		m_rtresultBuffer = (*m_pOptixContex)->createBufferFromGLBO(RT_BUFFER_OUTPUT, m_resultPbo);
		m_rtresultBuffer->setSize(m_width, m_height);
		m_rtresultBuffer->setFormat(RT_FORMAT_FLOAT4);
		
		(*m_pOptixContex)["construct_buffer"]->setBuffer(m_rtresultBuffer);
		
		(*m_pOptixContex)->setRayGenerationProgram(1, (*m_pOptixContex)->createProgramFromPTXFile(RAYTRACINGPATH, "reflection_request"));
		(*m_pOptixContex)->setExceptionProgram(1, (*m_pOptixContex)->createProgramFromPTXFile(RAYTRACINGPATH, "exception"));
		(*m_pOptixContex)->setMissProgram(1, (*m_pOptixContex)->createProgramFromPTXFile(TEXTUREPATH, "miss"));
		

	}
	catch (optix::Exception& e)
	{
		printf("%s\n", e.getErrorString().c_str());
		exit(1);
	}
}

void Constructor::render(glslShader & shader, textureManager& manager)
{
	if (primary_ray_e == m_pScene->getRayType())
	{
		m_construct_Fbo.begin();
		m_pScene->render(shader, manager, m_pNaviCam);
		m_construct_Fbo.end();
	}
	else if (reflected_ray_e == m_pScene->getRayType())
	{
		m_construct_Fbo.begin();
		m_pScene->render(shader, manager, m_pNaviCam);
		m_construct_Fbo.end();

		try {

			//nv::matrix4f modeViewInv = nv::matrix4f(pOriginCam->getModelViewInvMat());
			(*m_pOptixContex)["eye_pos"]->setFloat(m_pNaviCam->getCameraPos().x, m_pNaviCam->getCameraPos().y, m_pNaviCam->getCameraPos().z);
			(*m_pOptixContex)->launch(1, m_width, m_height);

				glPushAttrib(GL_PIXEL_MODE_BIT);
			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_resultPbo);
			glBindTexture(GL_TEXTURE_2D, m_resultTex);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height,
				GL_RGBA, GL_FLOAT, 0);
			glBindTexture(GL_TEXTURE_2D, 0);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
			glPopAttrib();

			

			
			glEnable(GL_TEXTURE_2D);
			BYTE *pTexture = NULL;
			pTexture = new BYTE[m_width * m_height * 3];
			memset(pTexture, 0, m_width * m_height * 3 * sizeof(BYTE));

			glBindTexture(GL_TEXTURE_2D, m_resultTex);//TexPosId   PboTex

			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pTexture);

			int w = m_width;
			int h = m_height;
			Fbo::SaveBMP("reflection_optix.bmp", pTexture, w, h);
			if (pTexture)
				delete[] pTexture;
			glBindTexture(GL_TEXTURE_2D, 0);//TexPosId   PboTex
			
		}
		catch (optix::Exception& e)
		{
			printf("%s\n", e.getErrorString().c_str());
			exit(1);
		}

	}
}
void Constructor::construct()
{
	m_constuctResource->map();
	mapConstruct(m_pNaviCam);
	m_optixColorResource->map();
	m_posBlendTex->map();
	m_gbufferGeoResource->map();
	m_gbufferNorResource->map();
	const int eocWidhth = ROWLARGER*m_width;
	const int eocHeight = ROWLARGER*m_height;
	edgeRendering(eocWidhth, eocHeight);
	cuda_Construct(m_width,m_height);

	
	m_gbufferGeoResource->unmap();
	m_gbufferNorResource->unmap();
	m_posBlendTex->unmap();

	m_optixColorResource->unmap();
	m_constuctResource->unMap();
	CHECK_ERRORS();

	m_constuctResource->generateTex();
	CHECK_ERRORS();
}
#include "Fbo.h"
#include "cuda.h"
#include "Camera.h"
#include "scene.h"
#ifndef CONSTURCTOR_H
#define CONSTURCTOR_H
extern void cuda_Construct(int,int);
extern void edgeRendering(int, int);
void mapConstruct(Camera * pReconstructCamer);
void construct_cudaInit();
class Constructor
{
public:
	Constructor() = default;
	Constructor(int w, int h) :m_width(w), m_height(h)
	{
		m_construct_Fbo = Fbo(3, w, h);

	}
	void init();
	inline void setNaveCam(Camera * pNaviCam)
	{
		m_pNaviCam = pNaviCam;
	}
	inline Fbo& getBuffer()
	{
		return m_construct_Fbo;
	}
	inline GLuint getReconstructTexture()
	{
		return m_getTex;
	}
	inline void setScene(Scene *p_scene)
	{
		m_pScene = p_scene;
	}
	void render(glslShader & shader, textureManager& manager);
	void construct();
	void optixInit();
	inline void setOptixColorTex(int optixColorTex, int optixWidth, int optixHeight)
	{
		m_optixColorTex = optixColorTex;
		m_optixWidth = optixWidth;
		m_optixHeight = optixHeight;
	}
	inline void setGbufferTex(int gbuferGeoTex, int gbufferNorTex)
	{
		m_gbufferGeoTex = gbuferGeoTex;
		m_gbufferNorTex = gbufferNorTex;
	}
	inline void setGbufferSize(int x, int y)
	{
		m_naviWidth = x;
		m_naviHeight = y;
	}
	inline void setBlendPosBuffer(Fbo * pFbo)
	{
		pPosBlendFbo = pFbo;
	}
	inline void setOptixContex(optix::Context* p)
	{
		m_pOptixContex = p;
	}
private:
	optix::Context* m_pOptixContex;
	Scene* m_pScene;
	GLuint m_getTex,m_optixColorTex;
	GLuint m_gbufferGeoTex, m_gbufferNorTex;
	int m_width, m_height,m_optixWidth,m_optixHeight;
	int m_naviWidth, m_naviHeight;
	Fbo m_construct_Fbo;
	Camera * m_pNaviCam;
	Fbo * pPosBlendFbo;
	CudaPboResource * m_constuctResource;
	CudaTexResourse * m_optixColorResource, *m_posBlendTex;
	CudaTexResourse * m_gbufferGeoResource, *m_gbufferNorResource;
	optix::TextureSampler m_rtTexture,m_normalTexture;
	optix::Buffer         m_rtresultBuffer;

	GLuint m_resultTex;
	GLuint m_resultPbo;

};
#endif
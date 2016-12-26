#include "Fbo.h"
#include "cuda.h"
#include "Camera.h"
#include "scene.h"
#ifndef CONSTURCTOR_H
#define CONSTURCTOR_H
extern void cuda_Construct(int,int);
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
	inline void setOptixColorTex(int optixColorTex, int optixWidth, int optixHeight)
	{
		m_optixColorTex = optixColorTex;
		m_optixWidth = optixWidth;
		m_optixHeight = optixHeight;
	}
private:
	Scene* m_pScene;
	GLuint m_getTex,m_optixColorTex;
	int m_width, m_height,m_optixWidth,m_optixHeight;
	Fbo m_construct_Fbo;
	Camera * m_pNaviCam;
	CudaPboResource * m_constuctResource;
	CudaTexResourse * m_optixColorResource;

};
#endif
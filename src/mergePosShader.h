#include "glslShader.h"
#ifndef MERGEPOSSHADER_H
#define MERGEPOSSHADER_H
class Fbo;
class Camera;

class MergePosShader :public glslShader
{
public:
	MergePosShader()
	{
		m_vertexFileName = "Shader/mergePos.vert";
		m_fragmentFileName = "Shader/mergePos.frag";
	
	}
	virtual void init();
	virtual void bindParemeter();
	virtual void begin();
	virtual void end();
	virtual void setMaterial(const GLMmaterial & meterial, textureManager & manager);

	inline void setGbuffer(int p)
	{
		gBufferPosTex = p;
	}
	inline void setOptixPosTex(int p)
	{
		m_optixWorldTex = p;
	}
	inline void setRightCamera(Camera * pCam)
	{
		m_pRightCam = pCam;
	}

	inline void setParemeter();
private:
	GLuint gbufferSlot, m_optixWorldSlot,m_rightModelviewSlot;
	GLuint gBufferPosTex, m_optixWorldTex;
	Camera * m_pRightCam;
};
#endif
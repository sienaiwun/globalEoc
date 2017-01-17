#include "glslShader.h"
#ifndef BLENDSHADER_H
#define BLENDSHADER_H
class Fbo;
class Camera;

class BlendShader :public glslShader
{
public:
	BlendShader()
	{
		m_vertexFileName = "Shader/blend.vert";
		m_fragmentFileName = "Shader/blend.frag";
	
	}
	virtual void init();
	virtual void bindParemeter();
	virtual void begin();
	virtual void end();
	virtual void setMaterial(const GLMmaterial & meterial, textureManager & manager);

	inline void setGbuffer(Fbo * p)
	{
		pGbuffer = p;
	}
	inline void setProgBuffer(Fbo * p)
	{
		pProgBuffer = p;
	}


	inline void setParemeter();
private:
	GLuint gbufferSlot, progSlot;
	Fbo *pGbuffer, *pProgBuffer;
};
#endif
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

	inline void setBuffer1(Fbo * p)
	{
		pFbo1 = p;
	}
	inline void setBuffer2(Fbo * p)
	{
		pFbo2 = p;
	}


	inline void setParemeter();
private:
	GLuint color1Slot, color2Slot;
	Fbo *pFbo1, *pFbo2;
};
#endif
#include "glslShader.h"
#ifndef SHOW_SHADER_H
#define SHOW_SHADER_H
class Fbo;
class Camera;

class ShowShader :public glslShader
{
public:
	ShowShader()
	{
		m_vertexFileName = "Shader/showShader.vert";
		m_fragmentFileName = "Shader/showShader.frag";
	
	}
	virtual void init();
	virtual void bindParemeter();
	virtual void begin();
	virtual void end();
	virtual void setMaterial(const GLMmaterial & meterial, textureManager & manager);

	inline void setTex(GLuint tex)
	{
		m_tex = tex;
	}
	inline void setBegin(nv::vec2f begin)
	{
		m_begin = begin;
	}
	inline void setEnd(nv::vec2f end)
	{
		m_end = end;
	}
	inline void setParemeter();
private:
	nv::vec2f  m_begin, m_end;
	GLuint color1Slot, color2Slot;
	GLuint m_tex;
	GLuint m_beginSlot, m_endSlot;
};
#endif
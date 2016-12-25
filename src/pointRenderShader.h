#include "glslShader.h"
#ifndef POINTRENDER_H
#define POINTRENDER_H
class Fbo;
class Camera;
class PointRenderShader :public glslShader
{
public:
	PointRenderShader()
	{
		m_vertexFileName = "Shader/pointShader.vert";
		m_fragmentFileName = "Shader/pointShader.frag";
	}
	virtual void init();
	virtual void bindParemeter();
	virtual void begin();
	virtual void end();
	virtual void setMaterial(const GLMmaterial & meterial, textureManager & manager);

	inline void setColorTex(GLuint tex)
	{
		m_color_tex = tex;
	}
	inline void setPositonTex(GLuint tex)
	{
		m_world_tex = tex;
	}
	inline void setColorTex2(GLuint tex)
	{
		m_color_tex2 = tex;
	}
	inline void setPositonTex2(GLuint tex)
	{
		m_world_tex2 = tex;
	}
	inline void setParemeter();
	inline void setCamera(Camera *p)
	{
		pRenderCam = p;
	}
private:
	Camera * pRenderCam;
	GLuint m_color_tex, m_world_tex, m_color_tex2, m_world_tex2;
	GLuint m_color_tex_slot, m_world_tex_slot,m_mvp_slot;
};
#endif
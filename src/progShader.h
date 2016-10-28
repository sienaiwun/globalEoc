#include "glslShader.h"
#ifndef PROGATESHADER_H
#define PROGATESHADER_H
class Fbo;
class Camera;

class ProgShader :public glslShader
{
public:
	ProgShader()
	{
		m_vertexFileName = "Shader/propagate.vert";
		m_fragmentFileName = "Shader/propagate.frag";
		m_pCamera = NULL;
	}
	virtual void init();
	virtual void bindParemeter();
	virtual void begin();
	virtual void end();
	virtual void setMaterial(const GLMmaterial & meterial, textureManager & manager);

	inline void setGbuffer(Fbo * p)
	{
		m_pfbo = p;
	}
	inline void setNormalTex(GLuint tex)
	{
		m_normalTex = tex;
	}
	inline void setPositonTex(GLuint tex)
	{
		m_posTex = tex;
	}
	inline void setColorTex(GLuint tex)
	{
		m_colorTex = tex;
	}
	inline void setRes(nv::vec2f res)
	{
		m_res = res;
	}
	inline void setCamera(Camera * pC)
	{
		m_pCamera = pC;

	}
	inline void setEdgeFbo(Fbo * pfbo)
	{
		m_pedgeFbo = pfbo;
	}
	inline void setParemeter();
private:
	GLuint m_normalTex, m_normalTexSlot;
	GLuint m_posTex, m_posTexSlot;
	GLuint m_colorTex, m_edgeTexSlot;
	GLuint m_res_slot, m_mvpSlot, m_mvpInv_slot;
	GLuint m_modelView_slot, m_modelViewInv_slot;
	GLuint m_cameraPos_slot;
	GLuint m_bbminSlot, m_bbmax_slot;
	Camera * m_pCamera;
	Fbo * m_pfbo;
	Fbo * m_pedgeFbo;
	float* m_mvp;
	float* m_modelView;
	nv::vec2f m_res;
	nv::vec3f m_cameraPos;
};
#endif
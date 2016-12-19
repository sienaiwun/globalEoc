#include "glslShader.h"
#include "Fbo.h"
#ifndef EOCVOLUMN_H
#define EOCVOLUMN_H
class EocVolumnShader :public glslShader
{
public:
	EocVolumnShader()
	{
		m_vertexFileName = "Shader/eocVolumn.vert";
		m_geometryFileName = "Shader/eocVolumn.geom";
		m_fragmentFileName = "Shader/eocVolumn.frag";
	}
	virtual void init();
	virtual void begin();
	virtual void end();
	virtual void setCamera(Camera * pCamera)
	{
		m_pCamera = pCamera;
	}
	virtual void setScene(Scene * pScene);
	virtual void setGeomtryIndex(int i);
	virtual void setMaterial(const GLMmaterial & meterial, textureManager & manager);
	inline void setEdgeFbo(Fbo * pFbo)
	{
		m_pEdgeFbo = pFbo;
	}
	inline void setGbuffer(Fbo * pFbo)
	{
		m_pGbuffer = pFbo;
	}
	void setPara();
	inline void setDiffuse(nv::vec3f color)
	{
		m_diffuseColor = color;
	}
	inline void setRes(nv::vec2f res)
	{
		m_resolution = res;
	}
	inline void setEocCamera(nv::vec3f pos)
	{
		m_eocCameraPos = pos;
	}
	inline void setIsVertival(bool v)
	{
		m_isVertical = v;
	}
	inline void setAssoTex(GLuint tex)
	{
		m_asso_tex = tex;
	}
private:

	GLuint m_vmpBinding;
	GLuint m_modelViewBinding;
	float* m_mvp;
	float* m_modelView;
	GLuint m_posTexSlot, m_pEdgeSlot;
	GLuint m_objectDiffuseBinding;
	GLuint m_cameraPosBinding;
	GLuint m_reselutionSlot;
	GLuint m_hasTex;
	GLuint m_objectId;
	GLuint m_reflectFactor;
	GLuint m_eocCamSlot,m_isVertical_slot;
	GLuint m_asso_tex_slot, m_asso_tex;
	Fbo * m_pGbuffer;
	Camera * m_pCamera;
	Fbo * m_pEdgeFbo;
	float m_is_vertical;
	nv::vec3f m_diffuseColor;
	nv::vec2f c;
	nv::vec3f m_cameraPos;
	nv::vec2f m_resolution;
	nv::vec3f m_eocCameraPos;
	bool m_isVertical;


};
#endif
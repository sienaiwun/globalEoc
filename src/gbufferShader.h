#include "glslShader.h"
#ifndef GBUFFERSHADER_H
#define GBUFFERSHADER_H
class GbufferShader :public glslShader
{
public:
	GbufferShader()
	{
		m_vertexFileName = "Shader/gBuffer.vert";
		m_fragmentFileName = "Shader/gBuffer.frag";
	}
	virtual void init();
	virtual void begin();
	virtual void end();
	virtual void setCamera(Camera * pCamera);
	virtual void setScene(Scene * pScene);
	virtual void setGeomtryIndex(int i);
	virtual void setMaterial(const GLMmaterial & meterial, textureManager & manager);
	inline void setDiffuse(nv::vec3f color)
	{
		m_diffuseColor = color;
	}
private:

	GLuint m_vmpBinding;
	GLuint m_modelViewBinding;
	float* m_mvp;
	float* m_modelView;
	GLuint m_objectTexBinding;
	GLuint m_objectDiffuseBinding;
	GLuint m_cameraPosBinding;
	GLuint m_lightPosBinding;
	GLuint m_hasTex;
	GLuint m_objectId;
	GLuint m_reflectFactor;

	nv::vec3f m_lightPos;
	nv::vec3f m_cameraPos;
	nv::vec3f m_diffuseColor;


};
#endif
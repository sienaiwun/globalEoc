#include "glslShader.h"
#ifndef TRANSPARENTSHADER_H
#define TRANSPARENTSHADER_H
class TransparentShader :public glslShader
{
public:
	TransparentShader()
	{
		m_vertexFileName = "Shader/transparentShader.vert";
		m_fragmentFileName = "Shader/transparentShader.frag";
	}
	virtual void init();
	virtual void begin();
	virtual void end();
	virtual void setCamera(Camera * pCamera);
	virtual void setScene(Scene * pScene);
	virtual void setGeomtryIndex(int i);
	virtual void setMaterial(const GLMmaterial & meterial, textureManager & manager);
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


};
#endif
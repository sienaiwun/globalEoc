#include "glslShader.h"
#ifndef TRANSPARENTFIXEDSHADER_H
#define TRANSPARENTFIXEDSHADER_H
class TransparentFixedShader :public glslShader
{
public:
	TransparentFixedShader()
	{
		m_vertexFileName = "Shader/transparentShader.vert";
		m_fragmentFileName = "Shader/transparentFixedShader.frag";
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
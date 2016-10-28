#include "eocVolumn.h"
#include "scene.h"

void EocVolumnShader::init()
{
	m_loader.loadShader(m_vertexFileName.c_str(), m_geometryFileName.c_str(), m_fragmentFileName.c_str());
	m_vmpBinding = m_loader.getUniform("MVP");
	m_cameraPosBinding = m_loader.getUniform("cameraPos");
	m_posTexSlot = m_loader.getUniform("posTex");
	m_pEdgeSlot = m_loader.getUniform("edgeTex");
	m_reselutionSlot = m_loader.getUniform("resolution");
	m_eocCamSlot = m_loader.getUniform("eocCameraPos");
}

void EocVolumnShader::setScene(Scene * pScene)
{
	
}
void EocVolumnShader::setGeomtryIndex(int i)
{
	

}
void EocVolumnShader::begin()
{
	m_loader.useShder();
	setPara();
}
void EocVolumnShader::end()
{
	m_loader.DisUse();

}

void EocVolumnShader::setMaterial(const GLMmaterial & material, textureManager & manager)
{
	//if (material.ambient_map[0] == 'a');
}
void EocVolumnShader::setPara()
{
	CHECK_ERRORS();
	m_mvp = m_pCamera->getMvpMat();
	m_modelView = m_pCamera->getModelViewMat();
	CHECK_ERRORS();
	glUniformMatrix4fv(m_vmpBinding, 1, GL_FALSE, m_mvp);
	CHECK_ERRORS();
	glUniform3f(m_cameraPosBinding, m_pCamera->getCameraPos().x, m_pCamera->getCameraPos().y, m_pCamera->getCameraPos().z);
	CHECK_ERRORS();
	glUniform2f(m_reselutionSlot, m_resolution.x, m_resolution.y);
	glUniform3f(m_eocCamSlot, m_eocCameraPos.x, m_eocCameraPos.y, m_eocCameraPos.z);
	CHECK_ERRORS();
	setShaderTex(m_posTexSlot, m_pGbuffer->getTexture(1)); 
	CHECK_ERRORS();
	setShaderTex(m_pEdgeSlot, m_pEdgeFbo->getTexture(0));

	resetTexId();

}
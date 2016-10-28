#include "newEdge.h"
#include "Camera.h"
#include "Fbo.h"
#include <assert.h>
void NewEdgeShader::init()
{
	m_loader.loadShader(m_vertexFileName.c_str(), 0, m_fragmentFileName.c_str());
	m_colorTexSlot = m_loader.getUniform("colorTex");
	m_posTexSlot = m_loader.getUniform("posTex");
	m_normalTexSlot = m_loader.getUniform("normalTex");
	
	
	m_res_slot = m_loader.getUniform("resolution");
	m_mvpSlot = m_loader.getUniform("MVP");
	m_mvpInv_slot = m_loader.getUniform("mvpInv");
	m_cameraPos_slot = m_loader.getUniform("cameraPos");
	m_modelView_slot = m_loader.getUniform("modeView");
	m_modelViewInv_slot = m_loader.getUniform("modeViewInv");
	m_projection_slot = m_loader.getUniform("projection");
	

}
void NewEdgeShader::setParemeter()
{
	m_colorTex = m_pfbo->getTexture(0);
	m_posTex = m_pfbo->getTexture(1);

	m_normalTex = m_pfbo->getTexture(2);
	
	assert(m_pCamera != NULL);
	m_cameraPos = m_pCamera->getCameraPos();
	m_mvp = m_pCamera->getMvpMat();
	m_modelView = m_pCamera->getModelViewMat();
	

} 
void NewEdgeShader::bindParemeter()
{
	setParemeter();
	
	m_cameraPos = m_pCamera->getCameraPos();
	m_mvp = m_pCamera->getMvpMat();
	m_modelView = m_pCamera->getModelViewMat();
	setShaderTex(m_colorTexSlot, m_colorTex);

	setShaderTex(m_posTexSlot, m_posTex);

	setShaderTex(m_normalTexSlot, m_normalTex);


	glUniform2f(m_res_slot, m_res.x, m_res.y);
	glUniformMatrix4fv(m_mvpSlot, 1, GL_FALSE, m_mvp);
	nv::matrix4f invMatrix = inverse(nv::matrix4f(m_mvp));
	glUniformMatrix4fv(m_mvpInv_slot, 1, GL_FALSE, invMatrix.get_value());
	glUniform3f(m_cameraPos_slot, m_cameraPos.x, m_cameraPos.y, m_cameraPos.z);

	glUniformMatrix4fv(m_modelView_slot, 1, GL_FALSE, m_modelView);
	nv::matrix4f invModelViewMatrix = inverse(nv::matrix4f(m_modelView));
	glUniformMatrix4fv(m_modelViewInv_slot, 1, GL_FALSE, invModelViewMatrix.get_value());
	glUniformMatrix4fv(m_projection_slot, 1, GL_FALSE, m_pCamera->getProjection());

}
void NewEdgeShader::setMaterial(const GLMmaterial & meterial, textureManager & manager)
{
	bindParemeter();
}
void NewEdgeShader::begin()
{
	m_loader.useShder();
	bindParemeter();
}
void NewEdgeShader::end()
{
	resetTexId();
	m_loader.DisUse();
}
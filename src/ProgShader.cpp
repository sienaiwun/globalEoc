#include "ProgShader.h"
#include "Camera.h"
#include "Fbo.h"
#include <assert.h>
void ProgShader::init()
{
	m_loader.loadShader(m_vertexFileName.c_str(), 0, m_fragmentFileName.c_str());
	m_edgeTexSlot = m_loader.getUniform("edgeTex");
	m_posTexSlot = m_loader.getUniform("posTex");

	m_normalTexSlot = m_loader.getUniform("normalTex");


	m_res_slot = m_loader.getUniform("resolution");
	m_mvpSlot = m_loader.getUniform("MVP");
	m_mvpInv_slot = m_loader.getUniform("mvpInv");
	m_cameraPos_slot = m_loader.getUniform("cameraPos");
	m_modelView_slot = m_loader.getUniform("modeView");
	m_modelViewInv_slot = m_loader.getUniform("modeViewInv");


}
void ProgShader::setParemeter()
{
	m_colorTex = m_pfbo->getTexture(0);
	m_posTex = m_pfbo->getTexture(1);

	m_normalTex = m_pfbo->getTexture(2);

	


}
void ProgShader::bindParemeter()
{
	setParemeter();

	setShaderTex(m_posTexSlot, m_posTex);
	setShaderTex(m_edgeTexSlot,m_pedgeFbo->getTexture(0));
	glUniform2f(m_res_slot, m_res.x, m_res.y);

}
void ProgShader::setMaterial(const GLMmaterial & meterial, textureManager & manager)
{
	bindParemeter();
}
void ProgShader::begin()
{
	m_loader.useShder();
	bindParemeter();
}
void ProgShader::end()
{
	resetTexId();
	m_loader.DisUse();
}
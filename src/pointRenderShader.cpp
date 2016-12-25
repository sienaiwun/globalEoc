#include "pointRenderShader.h"
#include "Camera.h"
#include "Fbo.h"
#include <assert.h>
void PointRenderShader::init()
{
	m_loader.loadShader(m_vertexFileName.c_str(), 0, m_fragmentFileName.c_str());
	m_color_tex_slot = m_loader.getUniform("colorTex");
	m_world_tex_slot = m_loader.getUniform("worldPosTex");
	m_mvp_slot = m_loader.getUniform("MVP");




}
void PointRenderShader::setParemeter()
{
	

}
void PointRenderShader::bindParemeter()
{
	setParemeter();
	setShaderTex(m_color_tex_slot, m_color_tex);
	setShaderTex(m_world_tex_slot, m_world_tex);
	glUniformMatrix4fv(m_mvp_slot, 1, GL_FALSE, pRenderCam->getMvpMat());
	
}
void PointRenderShader::setMaterial(const GLMmaterial & meterial, textureManager & manager)
{
	bindParemeter();
}
void PointRenderShader::begin()
{
	m_loader.useShder();
	bindParemeter();
}
void PointRenderShader::end()
{
	resetTexId();
	m_loader.DisUse();
}
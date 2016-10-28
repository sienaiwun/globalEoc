#include "showShader.h"
#include "Camera.h"
#include "Fbo.h"
#include <assert.h>
void ShowShader::init()
{
	m_loader.loadShader(m_vertexFileName.c_str(), 0, m_fragmentFileName.c_str());
	color1Slot = m_loader.getUniform("colorTex");
	m_beginSlot = m_loader.getUniform("begin");
	m_endSlot = m_loader.getUniform("end");

}
void ShowShader::setParemeter()
{
	
	

} 
void ShowShader::bindParemeter()
{
	setShaderTex(color1Slot, m_tex);
	glUniform2f(m_beginSlot, m_begin.x, m_begin.y);
	glUniform2f(m_endSlot, m_end.x, m_end.y);
}
void ShowShader::setMaterial(const GLMmaterial & meterial, textureManager & manager)
{
}
void ShowShader::begin()
{
	m_loader.useShder();
	bindParemeter();
}
void ShowShader::end()
{
	resetTexId();
	m_loader.DisUse();
}
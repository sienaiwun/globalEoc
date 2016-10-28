#include "blender.h"
#include "Camera.h"
#include "Fbo.h"
#include <assert.h>
void BlendShader::init()
{
	m_loader.loadShader(m_vertexFileName.c_str(), 0, m_fragmentFileName.c_str());
	color1Slot = m_loader.getUniform("color1Tex");
	color2Slot = m_loader.getUniform("color2Tex");
	

}
void BlendShader::setParemeter()
{
	
	

} 
void BlendShader::bindParemeter()
{
	
	setShaderTex(color1Slot, pFbo1->getTexture(0));
	setShaderTex(color2Slot, pFbo2->getTexture(0));
	

}
void BlendShader::setMaterial(const GLMmaterial & meterial, textureManager & manager)
{
}
void BlendShader::begin()
{
	m_loader.useShder();
	bindParemeter();
}
void BlendShader::end()
{
	resetTexId();
	m_loader.DisUse();
}
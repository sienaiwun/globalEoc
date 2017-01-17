#include "blender.h"
#include "Camera.h"
#include "Fbo.h"
#include <assert.h>
void BlendShader::init()
{
	m_loader.loadShader(m_vertexFileName.c_str(), 0, m_fragmentFileName.c_str());
	gbufferSlot = m_loader.getUniform("posTex");
	progSlot = m_loader.getUniform("progTex");
	

}
void BlendShader::setParemeter()
{
	
	

} 
void BlendShader::bindParemeter()
{
	
	setShaderTex(gbufferSlot, pGbuffer->getTexture(1));
	setShaderTex(progSlot, pProgBuffer->getTexture(0));
	

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
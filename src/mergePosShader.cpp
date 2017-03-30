#include "mergePosShader.h"
#include "Camera.h"
#include "Fbo.h"
#include <assert.h>
void MergePosShader::init()
{
	m_loader.loadShader(m_vertexFileName.c_str(), 0, m_fragmentFileName.c_str());
	gbufferSlot = m_loader.getUniform("posTex");
	m_optixWorldSlot = m_loader.getUniform("optixPosTex");
	

}
void MergePosShader::setParemeter()
{
	
	

} 
void MergePosShader::bindParemeter()
{
	
	setShaderTex(gbufferSlot, gBufferPosTex);
	setShaderTex(m_optixWorldSlot,m_optixWorldTex);
	

}
void MergePosShader::setMaterial(const GLMmaterial & meterial, textureManager & manager)
{
}
void MergePosShader::begin()
{
	m_loader.useShder();
	bindParemeter();
}
void MergePosShader::end()
{
	resetTexId();
	m_loader.DisUse();
}
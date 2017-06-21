#include "scene.h"
#include "textureManager.h"
#include "edgeShader.h"
#include "newEdge.h"
#include "gbufferShader.h"
#include "progShader.h"
#include "eocCamera.h"
#include "eocVolumn.h"
#include "fbo.h"
#include "blender.h"
#include "macro.h"
#include "rowCounter.h"
#include "mergePosShader.h"
#ifdef OPTIX
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
using namespace optix;
#include <cuda.h>
#include <cuda_runtime.h>

#endif


#ifndef EOC_H
#define EOC_H
class EOCrender
{
public:
	EOCrender();
	EOCrender(int w, int h);
	~EOCrender() = default;
	inline void setOriginCamera(Camera * pCamera)
	{
		pOriginCam = pCamera;
		m_eocRightCam.setOriginCamera(pCamera);
		m_eocTopCam.setOriginCamera(pCamera);
	}
	void render(textureManager & manager);
	inline int getWidth()
	{
		return m_width;
	}
	inline int getHeight()
	{
		return m_height;
	}
	inline void setScene(Scene *pScene)
	{
		m_pScene = pScene;
	}
	inline Fbo* getGbufferP()
	{
		return &m_gbufferFbo;
	}
	inline Fbo* getEdgeBufferP()
	{
		return &m_edgeFbo;
	}
	inline Fbo* getRenderFbo()
	{
		return &m_renderFbo;
	}
	inline void debugSwap()
	{
		m_debugSwap = !m_debugSwap;
	}
	inline Fbo* getRightOccludeFbo()
	{
		return &m_occludedRightBuffer;
	}
	inline Fbo* getTopOccludeFbo()
	{
		return &m_occludedTopBuffer;
	}
	inline Fbo* getRightEocBuffer()
	{
		return &m_gbufferRightEocFbo;
	}
	inline Fbo* getTopEocBuffer()
	{
		return &m_gbufferTopEocFbo;
	}
	inline GLuint getCudaTex()
	{
		return pCounter->getTex();
	}
	inline GLuint getCudaTopTex()
	{
		return pCounter->getTopTex();
	}
	inline EocCamera* getRightEocCamera()
	{
		return &m_eocRightCam;
	}
	inline EocCamera* getTopEocCamera()
	{
		return &m_eocTopCam;
	}
	inline int getOptixWidth()
	{
		return m_cudaTexWidth;
	}
	inline int getOptixHeight()
	{
		return m_cudaTexHeight;
	}
	inline nv::vec3f getRightND()
	{
		return m_eocRightCam.getD();
	}
	inline nv::vec3f getTopD()
	{
		return m_eocTopCam.getD();
	}
	inline Fbo getPosBlendFbo()
	{
		return m_posBlendFbo;
	}
	inline Fbo getMergePosFbo()
	{
		return m_margePosFbo;
	}
private:
	bool m_debugSwap;
	EocCamera m_eocRightCam;
	EocCamera m_eocTopCam;
	Camera * pOriginCam;  // todo 
	Camera * pNaviCam;
	NewEdgeShader m_edgeShader;
	GbufferShader m_gbufferShader;
	BlendShader m_blendShader;
	EocVolumnShader m_volumnShader;
	MergePosShader m_mergePosShader;
	ProgShader g_progShader;
	Scene * m_pScene, *m_pQuad;
	GLuint m_width, m_height, m_k;
	int m_total_pixel;
	GLuint m_head_pointer_texture, m_head_pointer_initializer, m_atomic_counter_buffer, m_fragment_storage_buffer, m_linked_list_texture, m_computerShader, dispatch_buffer;
	GLuint *m_data;
	Fbo m_renderFbo;
	Fbo m_edgeFbo;
	Fbo m_progFbo;
	Fbo m_posBlendFbo;
	Fbo m_margePosFbo;
	Fbo m_gbufferFbo;
	Fbo m_gbufferRightEocFbo;
	Fbo m_gbufferTopEocFbo;
	Fbo m_occludedRightBuffer;
	Fbo m_occludedTopBuffer;
	Fbo debugFbo;
	int m_cudaTexWidth, m_cudaTexHeight;
	RowCounter * pCounter;
#ifdef OPTIX
public:
	void initOptix();
	void optixTracing();
	inline GLuint getOptixTex()
	{
		return m_optixTex;
	}
	inline GLuint getOptixWorldTex()
	{
		return m_optixWorldTex;
	}
    (optix::Context*) getOptixContex()
	{
		return &m_rtContext;
	}
	void EOCEdgeRender();
private:
	optix::Context        m_rtContext;
	optix::Buffer         m_rtfinalBuffer;
	optix::Buffer         m_rtWorldBuffer;
	optix::TextureSampler m_rtTexture;
	GLuint m_optixPbo;
	GLuint m_optixTex;
	GLuint m_optixWorldPbo;
	GLuint m_optixWorldTex;

#endif
};
#endif
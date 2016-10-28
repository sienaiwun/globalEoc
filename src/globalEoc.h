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
	}
	void render(textureManager & manager);
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
	inline Fbo* getOccludeFbo()
	{
		return &m_occludedBuffer;
	}
	inline Fbo* getEocBuffer()
	{
		return &m_gbufferEocFbo;
	}
	inline GLuint getCudaTex()
	{
		return pCounter->getTex();
	}
	inline EocCamera* getEocCamera()
	{
		return &m_eocRightCam;
	}
private:
	bool m_debugSwap;
	EocCamera m_eocRightCam;
	Camera * pOriginCam;
	NewEdgeShader m_edgeShader;
	GbufferShader m_gbufferShader;
	BlendShader m_blendShader;
	EocVolumnShader m_volumnShader;
	ProgShader g_progShader;
	Scene * m_pScene, *m_pQuad;
	GLuint m_width, m_height, m_k;
	int m_total_pixel;
	GLuint m_head_pointer_texture, m_head_pointer_initializer, m_atomic_counter_buffer, m_fragment_storage_buffer, m_linked_list_texture, m_computerShader, dispatch_buffer;
	GLuint *m_data;
	Fbo m_renderFbo;
	Fbo m_edgeFbo;
	Fbo m_progFbo;
	Fbo m_gbufferFbo;
	Fbo m_gbufferEocFbo;
	Fbo m_occludedBuffer;
	Fbo debugFbo;

	RowCounter * pCounter;
#ifdef OPTIX
public:
	void initOptix();
	void optixTracing();
	inline GLuint getOptixTex()
	{
		return m_optixTex;
	}
private:
	optix::Context        m_rtContext;
	optix::Buffer         m_rtfinalBuffer;
	optix::TextureSampler m_rtTexture;
	GLuint m_optixPbo;
	GLuint m_optixTex;

#endif
};
#endif
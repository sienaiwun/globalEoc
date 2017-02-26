#ifndef ROWCOUNTER_H
#define ROWCOUNTER_H
#include "Fbo.h"
#include "cuda.h"
#include "Camera.h"
class RowCounter
{
public:
	RowCounter() = default;
	RowCounter(int w, int h, int k = 8);
	~RowCounter() = default;
	inline void setGbuffer(Fbo * pFbo)
	{
		pGbuffer = pFbo;
	}
	inline void setTopOccludorBuffer(Fbo * pFbo)
	{
		pTopOccluderFbo = pFbo;
	}
	inline void setRightOccludorBuffer(Fbo * pFbo)
	{
		pRightOccluderFbo = pFbo;
	}
	inline void setEdgeBuffer(Fbo * pFbo)
	{
		pEdgeFbo = pFbo;
	}
	inline GLuint getTex() 
	{
		return m_getTex;
	}
	inline GLuint getTopTex()
	{
		return m_getTopTex;
	}
	void refresh();
	void init();
	void render(Camera *pCamera, Camera *pEoc, Camera * pTopCamer);
private:
	const int m_width, m_height,m_k;
	GLuint m_getTex;
	GLuint m_getTopTex;
	Fbo * pGbuffer;
	Fbo * pRightOccluderFbo;
	Fbo * pTopOccluderFbo;
	Fbo * pEdgeFbo;
	GLuint m_head_pointer_texture, m_head_pointer_initializer, m_atomic_counter_buffer, m_fragment_storage_buffer;
	GLuint *m_data;
	GLuint m_test;
	CudaTexResourse *m_colorCudaTex, *m_edgeCudaTex, *m_occluderRightCudaTex, *m_posCudaTex, *m_normalCudaTex ,*m_occluderTopCudaTex;
	CudaPboResource *m_initArray, *m_OutTex, *m_topOutTex;

};
#endif

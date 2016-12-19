#include "rowCounter.h"
#include <algorithm>
#include "assert.h"

#define CHECK_ERRORS()         \
	do {                         \
	GLenum err = glGetError(); \
	if (err) {                                                       \
	printf( "GL Error %d at line %d of FILE %s\n", (int)err, __LINE__,__FILE__);       \
	exit(-1);                                                      \
				}                                                                \
				} while(0)


static GLuint getAtomicCounter(GLuint buffer)
{
	// read back total fragments that been shaded
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, buffer);
	GLuint *ptr = (GLuint *)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), GL_MAP_READ_BIT);
	GLuint fragsCount = *ptr;
	glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
	return fragsCount;
}
void RowCounter::init()
{
	assert(pGbuffer != NULL);
	assert(pEdgeFbo != NULL);
	assert(pRightOccluderFbo != NULL);
	m_colorCudaTex = new CudaTexResourse();
	m_colorCudaTex->set(pGbuffer->getTexture(0), m_width, m_height, color_t);
	m_colorCudaTex->init();
	m_edgeCudaTex = new  CudaTexResourse();
	m_edgeCudaTex->set(pEdgeFbo->getTexture(0), m_width, m_height, edgebuffer_t);
	m_edgeCudaTex->init();
	m_occluderRightCudaTex = new  CudaTexResourse();
	m_occluderRightCudaTex->set(pRightOccluderFbo->getTexture(0), m_width, m_height, occluderbuffer_t);
	m_occluderRightCudaTex->init();

	m_occluderTopCudaTex = new  CudaTexResourse();
	m_occluderTopCudaTex->set(pTopOccluderFbo->getTexture(0), m_width, m_height, occluderTopbuffer_t);
	m_occluderTopCudaTex->init();


	m_posCudaTex = new  CudaTexResourse();
	m_posCudaTex->set(pGbuffer->getTexture(1), m_width, m_height, pos_t);
	m_posCudaTex->init();

	m_initArray = new CudaPboResource();
	m_initArray->set(1, m_height, list_e);
	m_initArray->init();
	m_OutTex = new CudaPboResource();
	m_OutTex->set(ROWLARGER*m_width, m_height, float4_t);
	m_OutTex->init();
	m_topOutTex = new CudaPboResource();
	m_topOutTex->set(ROWLARGER*m_width, ROWLARGER*m_height, top_float4_t);
	m_topOutTex->init();
	cudaInit(m_height, m_width, m_k, ROWLARGER);
	m_getTex = m_OutTex->getTexture();
	m_getTopTex = m_topOutTex->getTexture();

}

RowCounter::RowCounter(int w, int h, int k) :m_width(w), m_height(h), m_k(k)
{
	pGbuffer = NULL;
	pEdgeFbo = NULL;
	pRightOccluderFbo = NULL;
	pTopOccluderFbo = NULL;

}
void RowCounter::render(Camera *pCamera, Camera * pEocCam, Camera * pTopCamera)
{
	m_OutTex->map();
	m_initArray->map();
	m_occluderRightCudaTex->map();
	m_occluderTopCudaTex->map();
	m_edgeCudaTex->map();
	m_colorCudaTex->map();
	m_posCudaTex->map();
	m_topOutTex->map();

	countRow(m_width, m_height, pCamera, pEocCam, pTopCamera);
	m_colorCudaTex->unmap();
	m_edgeCudaTex->unmap();
	m_occluderRightCudaTex->unmap();
	m_occluderTopCudaTex->unmap();
	m_posCudaTex->unmap();
	m_initArray->unMap();
	
#ifdef DEBUG
	m_initArray->generateTex();
#endif
	m_OutTex->unMap();
	m_topOutTex->unMap();
	m_topOutTex->generateTex();
	m_OutTex->generateTex();
	
}
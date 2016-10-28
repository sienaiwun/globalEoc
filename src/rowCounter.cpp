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
	assert(pOccluderFbo != NULL);
	m_colorCudaTex = new CudaTexResourse();
	m_colorCudaTex->set(pGbuffer->getTexture(0), m_width, m_height, color_t);
	m_colorCudaTex->init();
	m_edgeCudaTex = new  CudaTexResourse();
	m_edgeCudaTex->set(pEdgeFbo->getTexture(0), m_width, m_height, edgebuffer_t);
	m_edgeCudaTex->init();
	m_occluderCudaTex = new  CudaTexResourse();
	m_occluderCudaTex->set(pOccluderFbo->getTexture(0), m_width, m_height, occluderbuffer_t);
	m_occluderCudaTex->init();

	m_posCudaTex = new  CudaTexResourse();
	m_posCudaTex->set(pGbuffer->getTexture(1), m_width, m_height, pos_t);
	m_posCudaTex->init();

	m_initArray = new CudaPboResource();
	m_initArray->set(1, m_height, list_e);
	m_initArray->init();
	m_OutTex = new CudaPboResource();
	m_OutTex->set(ROWLARGER*m_width, m_height, float4_t);
	m_OutTex->init();
	cudaInit(m_height, m_width, m_k, ROWLARGER);
	m_getTex = m_OutTex->getTexture();

}

RowCounter::RowCounter(int w, int h, int k) :m_width(w), m_height(h), m_k(k)
{
	pGbuffer = NULL;
	pEdgeFbo = NULL;
	pOccluderFbo = NULL;

}
void RowCounter::render(Camera *pCamera, Camera * pEocCam)
{
	m_OutTex->map();
	m_initArray->map();
	m_occluderCudaTex->map();
	m_edgeCudaTex->map();
	m_colorCudaTex->map();
	m_posCudaTex->map();

	countRow(m_width, m_height, pCamera, pEocCam);
	m_colorCudaTex->unmap();
	m_edgeCudaTex->unmap();
	m_occluderCudaTex->unmap();
	m_posCudaTex->unmap();
	m_initArray->unMap();
	
#ifdef DEBUG
	m_initArray->generateTex();
#endif
	m_OutTex->unMap();
	m_OutTex->generateTex();
	
}
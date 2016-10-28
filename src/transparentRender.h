#include "scene.h"
#include "textureManager.h"
#include "transparentShader.h"
#include "fbo.h"
#ifndef OITRENDER_H
#define OITRENDER_H
class OITrender
{
public:
	OITrender(int w, int h, int k);
	~OITrender();
	void render(Camera * pCamera,textureManager & manager);
	inline void setScene(Scene *pScene)
	{
		m_pScene = pScene;
	}
	inline Fbo& getFbo()
	{
		return m_renderFbo;
	}
	inline GLuint getRenderImage()
	{
		return m_renderFbo.getTexture(0);
	}
private:
	TransparentShader m_oitShader;
	Scene * m_pScene;
	GLuint m_width, m_height, m_k;
	int m_total_pixel;
	GLuint m_head_pointer_texture, m_head_pointer_initializer, m_atomic_counter_buffer, m_fragment_storage_buffer, m_linked_list_texture, m_computerShader, dispatch_buffer;
	GLuint *m_data;
	GLuint m_test;
	Fbo m_renderFbo;
};
#endif
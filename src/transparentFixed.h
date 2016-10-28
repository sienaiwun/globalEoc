#include "scene.h"
#include "textureManager.h"
#include "transparentFixedShader.h"
#include "fbo.h"
#ifndef OITRENDERFIXED_H
#define OITRENDERFIXED_H
class OITFixedRender
{
public:
	OITFixedRender(int w, int h, int k);
	~OITFixedRender();
	void render(Camera * pCamera, textureManager & manager);
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
	TransparentFixedShader m_oitShader;
	Scene * m_pScene;
	GLuint m_width, m_height, m_k;
	int m_total_pixel;
	GLuint m_head_pointer_texture, m_head_pointer_initializer, m_atomic_counter_buffer, m_fragment_storage_buffer, m_linked_list_texture, m_computerShader, dispatch_buffer;
	GLuint m_atomic_counter_array_buffer_texture, m_atomic_counter_array_buffer;
	GLuint *m_data;
	Fbo m_renderFbo;
};
#endif
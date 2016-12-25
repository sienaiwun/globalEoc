#include "Fbo.h"
#include "pointRenderShader.h"
class Camera;
class PointRender
{
public:
	PointRender() = default;
	PointRender(int w, int h) :m_width(w), m_height(h)
	{
	}
	void init();
	inline void setColorTex(int tex)
	{
		m_pointRenderShader.setColorTex(tex);
	}
	inline void setColorTex2(int tex)
	{
		m_pointRenderShader.setColorTex2(tex);
	}
	inline void setWorldTex(int tex)
	{
		m_pointRenderShader.setPositonTex(tex);
	}
	inline void setWorldTex2(int tex)
	{
		m_pointRenderShader.setPositonTex2(tex);
	}
	inline void setCamera(Camera * p)
	{
		m_pointRenderShader.setCamera(p);
	}
	void render(bool clear = true);
	inline int getPointRendering()
	{
		return m_pointRenderBuffer.getTexture(0);
	}
private:
	int m_width, m_height;
	Fbo m_pointRenderBuffer;
	PointRenderShader m_pointRenderShader;
	GLuint m_renderList;

};
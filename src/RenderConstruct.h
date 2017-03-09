#include "Fbo.h"
#include "cuda.h"
#include "Camera.h"
#include "scene.h"
class RenderConstruct
{
public:
	RenderConstruct() = default;
	RenderConstruct(GLuint optixColorTex, GLuint optixPosTex)
	{
		m_color_slot = optixColorTex;
		m_pos_slot = optixPosTex;
		m_pointPositonBuffer = 0;
		m_colorBuffer = 0;
	}
	inline void setSize(int w, int h)
	{
		m_width = w;
		m_height = h;
	}
	void Init();
	void build();
	
private:
	void mapToBuffer();
	GLuint m_optixColorTex, m_optixPosTex;
	int m_color_slot, m_pos_slot;
	int m_width, m_height;
	GLuint m_renderList;
	float * m_pointPositonBuffer;
	float * m_colorBuffer;
	
};
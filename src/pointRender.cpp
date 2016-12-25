#include "pointRender.h"
void PointRender::init()
{
	m_pointRenderShader.init();
	m_renderList = glGenLists(1);

	glNewList(m_renderList, GL_COMPILE);
	glPointSize(1);
	glBegin(GL_POINTS);
	for (int x =0; x < m_width; x++)
	{
		for (int y = 0; y < m_height; y++)
		{
			nv::vec3f value = nv::vec3f((x + 0.5f) / m_width, (y + 0.5f) / m_height, 1);
			glVertex3fv((float*)&value);
		}
	}
	glEnd();
	glEndList();
}
void PointRender::render(bool clear)
{
		CHECK_ERRORS();
		m_pointRenderShader.begin();
		glCallList(m_renderList);
		m_pointRenderShader.end();

}
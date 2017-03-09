#include "RenderConstruct.h"

struct Point
{
	nv::vec3f m_pos;
	nv::vec3f m_color;
	Point() = default;
	int isValid;
	Point(nv::vec3f pos, nv::vec3f color) :m_pos(pos), m_color(color)
	{
		if (length(pos) < 1)
			isValid = false;
		else
			isValid = true;
	}
};
void RenderConstruct::mapToBuffer()
{
	assert(m_width*m_height > 0);
	if (m_pointPositonBuffer)
	{
		delete m_pointPositonBuffer;
		m_pointPositonBuffer = 0;
	}
		if (m_colorBuffer)
	{
		delete m_colorBuffer;
		m_colorBuffer = 0;
	}
	glEnable(GL_TEXTURE_2D);
	m_pointPositonBuffer = new float[m_width*m_height * 3];
	memset(m_pointPositonBuffer, 0, m_width*m_height * 3 * sizeof(float));

	m_colorBuffer = new float[m_width*m_height * 3];
	memset(m_colorBuffer, 0, m_width*m_height * 3 * sizeof(float));


	glBindTexture(GL_TEXTURE_2D,m_optixPosTex);//TexPosId   PboTex
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, m_pointPositonBuffer);
	glBindTexture(GL_TEXTURE_2D, 0);//TexPosId   PboTex
	
	glBindTexture(GL_TEXTURE_2D, m_optixColorTex);//TexPosId   PboTex
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, m_colorBuffer);
	glBindTexture(GL_TEXTURE_2D, 0);//TexPosId   PboTex
}
void make_quad(Point& point1, Point& point2, Point& point3,Point& point4)
{
}
void makeTriangle(Point& point1, Point& point2, Point& point3)
{

}
void RenderConstruct::build()
{
	mapToBuffer();

	if (m_renderList)
		glDeleteLists(m_renderList, 1);
	m_renderList = glGenLists(1);
	glBegin(GL_TRIANGLES);
	for (int y = 0; y < m_height - 1; y++)	
	{
		for (int x = 0; x < m_width - 1; x++)
		{
			Point point1, point2, point3, point4;
			{
				nv::vec3f position = nv::vec3f(&m_pointPositonBuffer[3 * (y*m_width + x)]);
				nv::vec3f color = nv::vec3f(&m_colorBuffer[3 * (y*m_width + x)]);
				point1 = Point(position, color);
			}
			{
				int x1 = x + 1;
				int y1 = y + 1;
				nv::vec3f position = nv::vec3f(&m_pointPositonBuffer[3 * (y1*m_width + x1)]);
				nv::vec3f color = nv::vec3f(&m_colorBuffer[3 * (y1*m_width + x1)]);
				point2 = Point(position, color);
			}
			{
				int x2 = x + 1;
				int y2 = y;
				nv::vec3f position = nv::vec3f(&m_pointPositonBuffer[3 * (y2*m_width + x2)]);
				nv::vec3f color = nv::vec3f(&m_colorBuffer[3 * (y2*m_width + x2)]);
				point3 = Point(position, color);
			}
			{
				int x3 = x;
				int y3 = y + 1;
				nv::vec3f position = nv::vec3f(&m_pointPositonBuffer[3 * (y3*m_width + x3)]);
				nv::vec3f color = nv::vec3f(&m_colorBuffer[3 * (y3*m_width + x3)]);
				point3 = Point(position, color);
			}
			make_quad (point1, point2, point3, point4);
		}
	}
	glEndList();

}	
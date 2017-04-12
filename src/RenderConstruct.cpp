#include "RenderConstruct.h"

struct Point
{
	nv::vec3f m_pos;
	nv::vec3f m_color;
	int m_x, m_y;
	Point() = default;
	int isValid;
	Point(nv::vec3f pos, nv::vec3f color, int x, int y) :m_pos(pos), m_color(color), m_x(x), m_y(y)
	{
		if (length(pos) < 1)
			isValid = false;
		else
			isValid = true;
	}
	Point(nv::vec3f pos, nv::vec3f color, nv::vec2i cord) :m_pos(pos), m_color(color), m_x(cord.x), m_y(cord.y)
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
		m_pointPositonBuffer = new float[m_width*m_height * 3];
	memset(m_pointPositonBuffer, 0, m_width*m_height * 3 * sizeof(float));

	m_colorBuffer = new float[m_width*m_height * 3];
	memset(m_colorBuffer, 0, m_width*m_height * 3 * sizeof(float));

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,m_optixPosTex);//TexPosId   PboTex
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, m_pointPositonBuffer);
	glBindTexture(GL_TEXTURE_2D, 0);//TexPosId   PboTex
	
	glBindTexture(GL_TEXTURE_2D, m_optixColorTex);//TexPosId   PboTex
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, m_colorBuffer);
	glBindTexture(GL_TEXTURE_2D, 0);//TexPosId   PboTex
}
void renderLine(Point& end1, Point& end2)
{
	float dis = nv::length(end1.m_pos - end2.m_pos);
	if (end1.isValid&&end2.isValid&&dis<12)
	{
		glVertex3fv((float*)&end1.m_pos);
		glNormal3fv((float*)&end1.m_color);
		glVertex3fv((float*)&end2.m_pos);
		glNormal3fv((float*)&end2.m_color);
	}
}
void renderPoint(Point& point)
{
	if (point.isValid)
	{
		glVertex3fv((float*)&point.m_pos);
		glNormal3fv((float*)&point.m_color);
	}
}
nv::vec3f getIntersection(nv::vec3f pos, nv::vec3f normal, nv::vec3f& camera, nv::vec3f& lookAtPos)
{
	nv::vec3f toPlane = pos - camera;
	nv::vec3f toLookAt = lookAtPos - camera;
	float projToPlane = dot(toPlane, normal);
	float projToLookAt = dot(toLookAt, normal);
	nv::vec3f predict = camera + projToPlane*toLookAt / projToLookAt;
	return predict;
}
nv::vec3f plateInterpolation(nv::vec3f pos, nv::vec3f dir, nv::vec2f ndc, nv::vec3f cameraPos,int width,int height,nv::matrix4f& modelViewInv,nv::matrix4f& projInv)
{
	nv::vec2f NDC = nv::vec2f(ndc.x / width, ndc.y / height) * 2 - 1.0;
	nv::vec4f temp = modelViewInv* projInv* nv::vec4f(NDC.x, NDC.y, 0.0f, 1.0f);
	temp = temp / temp.w;
	nv::vec3f predictNear = nv::vec3f(temp.x, temp.y, temp.z);
	nv::vec3f predictPos = getIntersection(pos, dir, cameraPos, predictNear);
	return predictPos;
}
void renderQuad(Point& left, Point& right, nv::vec3f cameraPos, int w, int h, nv::matrix4f& modelViewInv, nv::matrix4f& projInv)
{
	float dis = nv::length(left.m_pos - right.m_pos);
#define OFFSET -0.005
	if (left.isValid&&right.isValid&&dis < 12)
	{
		{
		Point currentPoint = left;
		nv::vec3f dir = normalize(cameraPos - currentPoint.m_pos);
		nv::vec3f topLeftPos = plateInterpolation(currentPoint.m_pos, dir, nv::vec2f(currentPoint.m_x + 0.5, currentPoint.m_y + 1 - OFFSET), cameraPos, w, h, modelViewInv, projInv);
		glVertex3fv((float*)&topLeftPos);
		glNormal3fv((float*)&currentPoint.m_color);
		nv::vec3f ButLeftPos = plateInterpolation(currentPoint.m_pos, dir, nv::vec2f(currentPoint.m_x + 0.5, currentPoint.m_y + OFFSET), cameraPos, w, h, modelViewInv, projInv);
		glVertex3fv((float*)&ButLeftPos);
		glNormal3fv((float*)&currentPoint.m_color);
		}
		{
		Point currentPoint = right;
		nv::vec3f dir = normalize(cameraPos - currentPoint.m_pos);
		nv::vec3f ButLeftPos = plateInterpolation(currentPoint.m_pos, dir, nv::vec2f(currentPoint.m_x + 0.5, currentPoint.m_y + OFFSET), cameraPos, w, h, modelViewInv, projInv);
		glVertex3fv((float*)&ButLeftPos);
		glNormal3fv((float*)&currentPoint.m_color);
		 
		nv::vec3f topLeftPos = plateInterpolation(currentPoint.m_pos, dir, nv::vec2f(currentPoint.m_x + 0.5, currentPoint.m_y + 1 - OFFSET), cameraPos, w, h, modelViewInv, projInv);
		glVertex3fv((float*)&topLeftPos);
		glNormal3fv((float*)&currentPoint.m_color);
		
		}

	}

}
void make_quad(Point& point1, Point& rightTop, Point& right, Point& top, int w, int h,nv::matrix4f& modelViewInv, nv::matrix4f& projInv, nv::vec3f cameraPos)
{
	//renderLine(point1, right);
	//renderPoint(point1);
	//renderPoint(point2);
	//renderPoint(point3);
	//renderPoint(point4);
	renderQuad(point1,right,cameraPos,w,h,modelViewInv,projInv);
}
void makeTriangle(Point& point1, Point& point2, Point& point3)
{
	
}
void drawRay(nv::vec3f pos, nv::vec3f dir,nv::vec3f color)
{
	glBegin(GL_LINES);
	glVertex3fv((float*)&pos);
	glNormal3fv((float*)&color);
	glVertex3fv((float*)&(pos+ 155*dir));;
	glNormal3fv((float*)&color);
	glEnd();

}
 nv::vec2i  RenderConstruct::toImageCord(nv::vec3f pos)
{
	nv::vec4f temp = nv::matrix4f(p_construcCam->getMvpMat())* nv::vec4f(pos, 1.0f);
	temp = temp / temp.w;
	temp.x = temp.x * 0.5 + 0.5;
	temp.y = temp.y * 0.5 + 0.5;
	return nv::vec2i((temp.x*m_width), (temp.y*m_height));
}
void RenderConstruct::render(glslShader & shader, Camera * pCamera)
{
	glPointSize(3);
	glLineWidth(4);
	shader.begin();
	CHECK_ERRORS();
	CHECK_ERRORS();
	shader.setCamera(pCamera);
	renderSamples();
	CHECK_ERRORS();
	shader.end();
	CHECK_ERRORS();
}
void RenderConstruct::renderSamples()
{
	if (m_renderList)
		glCallList(m_renderList);
	drawRay(nv::vec3f(7.501300, -11.379437, -53.618397), nv::vec3f(-0.982115, -0.121992, 0.143415), nv::vec3f(0, 1, 0));// (x != 665 || y != 470)
	drawRay(nv::vec3f(7.503413, -11.379437, -53.611973), nv::vec3f(-0.981129, -0.122128, 0.149905), nv::vec3f(1, 0, 0));
}

void RenderConstruct::build()
{
	mapToBuffer();

	const nv::matrix4f modelView = nv::matrix4f(p_construcCam->getModelViewMat());
	const nv::matrix4f proj = nv::matrix4f(p_construcCam->getProjection());
	nv::matrix4f modelViewInv = inverse(modelView);
	nv::matrix4f projInv = inverse(proj);

	if (m_renderList)
		glDeleteLists(m_renderList, 1);
	m_renderList = glGenLists(1);
	glNewList(m_renderList, GL_COMPILE);
	//
	int w = m_width, h = m_height;
	nv::vec3f camPos = p_construcCam->getCameraPos();
	glBegin(GL_QUADS);
	for (int y = 0; y < m_height - 1; y++)	
	{
		for (int x = 0; x < m_width - 1; x++)
		//int x = 123, y = 123;
		{
			Point point1, rightTop, right, top;
			{
				nv::vec3f position = nv::vec3f(&m_pointPositonBuffer[3 * (y*m_width + x)]);
				nv::vec3f color = nv::vec3f(&m_colorBuffer[3 * (y*m_width + x)]);
				nv::vec2i cord = toImageCord(position);
				/*if ((cord.x != x && cord.x  > 1 && x < 1023) || (cord.y != y && cord.y > 1 && y <1023))
				{
					printf("aaa");
				}*/
				point1 = Point(position, color, cord.x, cord.y);
			}
			{
				int x1 = x + 1;
				int y1 = y + 1;
				nv::vec3f position = nv::vec3f(&m_pointPositonBuffer[3 * (y1*m_width + x1)]);
				nv::vec3f color = nv::vec3f(&m_colorBuffer[3 * (y1*m_width + x1)]);
				nv::vec2i cord = toImageCord(position);
				
				rightTop = Point(position, color, cord.x, cord.y);
			}
			{
				int x2 = x + 1;
				int y2 = y;
				nv::vec3f position = nv::vec3f(&m_pointPositonBuffer[3 * (y2*m_width + x2)]);
				nv::vec3f color = nv::vec3f(&m_colorBuffer[3 * (y2*m_width + x2)]);
				nv::vec2i cord = toImageCord(position);
				right = Point(position, color, cord.x, cord.y);
			}
			{
				int x3 = x;
				int y3 = y + 1;
				nv::vec3f position = nv::vec3f(&m_pointPositonBuffer[3 * (y3*m_width + x3)]);
				nv::vec3f color = nv::vec3f(&m_colorBuffer[3 * (y3*m_width + x3)]);
				nv::vec2i cord = toImageCord(position);
				top = Point(position, color, cord.x, cord.y);
			}
			make_quad(point1, rightTop, right, top, m_width, m_height, modelViewInv, projInv, camPos);
		}
	}
	glEnd();
	glEndList();

}	
#include <GL/glew.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include "Camera.h"
#include "Fbo.h"
#include "macro.h"
#include "gbuffershader.h"
#include "teapot.h"
#include "time.h"
#include "transparentRender.h"
#include "transparentFixed.h"
#include "globalEoc.h".
#include "box.h"
#include "showShader.h"
Timer g_time;
Camera g_Camera;
GbufferShader g_bufferShader;
ShowShader g_showShader;
Scene* g_scene;
EOCrender * pEoc;
textureManager texManager("");
bool drawFps = true;
OITrender *g_render;


void drawTex(GLuint mapId, bool addition = false, nv::vec2f beginPoint = nv::vec2f(0, 0), nv::vec2f endPoint = nv::vec2f(1, 1))
{
	g_showShader.setBegin(beginPoint);
	g_showShader.setEnd(endPoint);
	g_showShader.setTex(mapId);
	myGeometry::drawQuad(g_showShader, addition);
}

void key(unsigned char k, int x, int y)
{
	k = (unsigned char)tolower(k);
	switch (k)
	{
	case 'p':
		g_Camera.printToFile(g_scene->getCameraFile());
		break;
	case ' ':
		g_Camera.moveTo(2,20);
		break;
	case 't':
		pEoc->debugSwap();
		break;
	case '=':
		pEoc->getEocCamera()->addToOrigin(0.2);
		break;
	case '-':
		pEoc->getEocCamera()->addToOrigin(-0.2);
		break;
	}
	glutPostRedisplay();
}

void Reshape(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 0.01, 1000.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void externWglewInit();
void Init()
{
	glewInit();
	if (!glewIsSupported(
		"GL_VERSION_2_0 "
		"GL_ARB_vertex_program "
		"GL_ARB_fragment_program "
		"GL_ARB_texture_float "
		"GL_NV_gpu_program4 " // include GL_NV_geometry_program4
		"GL_ARB_texture_rectangle "
		))
	{
		printf("Unable to load extension()s:\n  GL_ARB_vertex_program\n  GL_ARB_fragment_program\n"
			"  GL_ARB_texture_float\n  GL_NV_gpu_program4\n  GL_ARB_texture_rectangle\n  OpenGL Version 2.0\nExiting...\n");
		exit(-1);
	}
	glEnable(GL_DEPTH_TEST);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)1.0, 0.01, 1000.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	externWglewInit();
	//g_render = new OITrender(SCREEN_WIDTH, SCREEN_HEIGHT, 20);

	pEoc = new EOCrender(SCREEN_WIDTH, SCREEN_HEIGHT);
	pEoc->setOriginCamera(&g_Camera);
	
	g_scene = new boxScene();
	g_scene->LoadCamera(&g_Camera);

	g_showShader.init();
	g_bufferShader.init();
	pEoc->setScene(g_scene);

	myGeometry::setTexManager(&texManager);
#ifdef OPTIX
	pEoc->initOptix();
#endif
}




void Display()
{
	CHECK_ERRORS();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	g_Camera.cameraControl();
	//g_scene->render(g_bufferShader, texManager, &g_Camera);            // diffuse rendering
	
	pEoc->render(texManager);

	//drawTex(pEoc->getCudaTex(), true, nv::vec2f(0.0, 0.0), nv::vec2f(0.65, 1.0));
	drawTex(pEoc->getOptixTex(), true, nv::vec2f(0.0, 0.0), nv::vec2f(0.65, 1.0));
	drawTex(pEoc->getGbufferP()->getTexture(0), true, nv::vec2f(0.75, 0.75));
	drawTex(pEoc->getOccludeFbo()->getTexture(0), true, nv::vec2f(0.75, 0.00), nv::vec2f(1, 0.25));
	drawTex(pEoc->getEocBuffer()->getTexture(0), true, nv::vec2f(0.75, 0.25), nv::vec2f(1, 0.50));
	
	if (drawFps ) {
		static char fps_text[32];
		float fps = g_time.getFps();
		sprintf(fps_text, "fps: %6.1f", fps);
		drawText(fps_text, 10.0f, 10.0f, GLUT_BITMAP_8_BY_13);
	}
	glutSwapBuffers();
	glutReportErrors();
}
void idle()
{
	g_Camera.Update();
	glutPostRedisplay();
}

int main(int argc, char** argv)
{
	freopen("stdout.txt", "w", stdout);
	freopen("stderr.txt", "w", stderr);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
	glutCreateWindow("���");
	Init();
	glutDisplayFunc(Display);
	glutReshapeFunc(Reshape);
	glutKeyboardFunc(key);
	glutIdleFunc(idle);
	glutMainLoop();
	return 0;
}


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
#include "boxReflected.h"
#include "showShader.h"
#include "Constructor.h"
#include "pointRender.h"
#include "RenderConstruct.h"
#include "greconstructShader.h"
Timer g_time;
static Camera g_Camera;
static Camera g_navi_Cam;
GbufferShader g_bufferShader;
ShowShader g_showShader;
Scene* g_scene;
EOCrender * pEoc;
textureManager texManager("");
bool drawFps = true;
OITrender *g_render;
bool _isNaviCam = true;
static Constructor g_Consturctor;
PointRender g_pointRender;
PointRender g_OptixPointRender;
Fbo g_naviGbuffer;
RenderConstruct g_renderConstruct;
RenderConstruct g_renderGbufferConstruct;
GReconstructShader g_reconstructShader;
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
		if (_isNaviCam)
		{
			g_navi_Cam.printToFile(g_scene->getNaviCamerFile());
		}
		else
		{
			g_Camera.printToFile(g_scene->getCameraFile());
		}
		break;
	case ' ':
		g_Camera.moveTo(2,20);
		break;
	case 't':
		pEoc->debugSwap();
		break;
	case '=':
		pEoc->getRightEocCamera()->addToOrigin(0.2);
		break;
	case '-':
		pEoc->getRightEocCamera()->addToOrigin(-0.2);
		break;
	case '9':
		pEoc->getTopEocCamera()->addToOrigin(0.2);
		break;
	case '0':
		pEoc->getTopEocCamera()->addToOrigin(-0.2);
		break;
	case '	'://tab
		_isNaviCam = !_isNaviCam;
		break;
	}
	glutPostRedisplay();
}

void Reshape(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
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
	//gluPerspective(60.0, (GLfloat)1.0, 0.01, 1000.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	externWglewInit();
	//g_render = new OITrender(SCREEN_WIDTH, SCREEN_HEIGHT, 20);

	pEoc = new EOCrender(SCREEN_WIDTH, SCREEN_HEIGHT);//同时进行初始化
	pEoc->setOriginCamera(&g_Camera);
	
	g_scene = new boxReflectedScene();
	g_scene->LoadCamera(&g_Camera);
	g_scene->LoadNaviCam(&g_navi_Cam);

	g_showShader.init();
	g_bufferShader.init();
	pEoc->setScene(g_scene);

	myGeometry::setTexManager(&texManager);
#ifdef OPTIX
	pEoc->initOptix();
#endif

	g_Camera.cameraControl();
	g_navi_Cam.cameraControl();
	g_pointRender = PointRender(SCREEN_WIDTH, SCREEN_HEIGHT	);
	g_pointRender.init();

	g_OptixPointRender = PointRender(SCREEN_WIDTH, SCREEN_HEIGHT);
	g_OptixPointRender.init();

	g_naviGbuffer = Fbo(3, SCREEN_WIDTH, SCREEN_HEIGHT);
	g_naviGbuffer.init();

	g_Consturctor = Constructor(pEoc->getWidth(), pEoc->getHeight());
	g_Consturctor.setNaveCam(&g_navi_Cam);
	g_Consturctor.setScene(g_scene);
	g_Consturctor.setOptixColorTex(pEoc->getOptixTex(), pEoc->getOptixWidth(), pEoc->getOptixHeight());
	g_Consturctor.setGbufferTex(g_naviGbuffer.getTexture(1), g_naviGbuffer.getTexture(2));
	g_Consturctor.setGbufferSize(g_naviGbuffer.getTexState().getWidth(), g_naviGbuffer.getTexState().getHeight());
	g_Consturctor.setBlendPosBuffer(&pEoc->getPosBlendFbo());
	g_Consturctor.init();

	
#ifdef OPTIX
	g_Consturctor.setOptixContex(pEoc->getOptixContex());
	g_Consturctor.optixInit();
	(*pEoc->getOptixContex())->setStackSize(2048);
	(*pEoc->getOptixContex())->validate();

	
#endif
	pEoc->render(texManager);

	g_reconstructShader.init();
	g_renderConstruct = RenderConstruct(pEoc->getOptixTex(), pEoc->getOptixWorldTex());
	g_renderConstruct.setSize(pEoc->getOptixWidth(), pEoc->getOptixHeight());

	
	/*
	g_renderConstruct.setConstructCam(pEoc->getRightEocCamera()->getEocCameraP());
	g_renderConstruct.build();

	g_renderGbufferConstruct = RenderConstruct(pEoc->getGbufferP()->getTexture(0), pEoc->getGbufferP()->getTexture(1));
	g_renderGbufferConstruct.setSize(pEoc->getWidth(), pEoc->getHeight());
	g_renderGbufferConstruct.setConstructCam(&g_Camera);
	g_renderGbufferConstruct.build();
	*/

}




void Display()
{
	CHECK_ERRORS();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if (!_isNaviCam)
	{
		g_Camera.cameraControl();
	}
	else
	{
		g_navi_Cam.cameraControl();
	}
	
	g_naviGbuffer.begin();
	g_scene->render(g_bufferShader, texManager, &g_navi_Cam);
	//g_renderConstruct.render(g_reconstructShader, &g_navi_Cam);
	//g_renderGbufferConstruct.render(g_reconstructShader, &g_navi_Cam);
	//g_naviGbuffer.SaveBMP("real.bmp",0);
	g_naviGbuffer.end();
	drawTex(g_naviGbuffer.getTexture(0), true, nv::vec2f(0., 0.6), nv::vec2f(0.4, 1.0));


	//drawTex(pEoc->getCudaTex(), true, nv::vec2f(0.0, 0.0), nv::vec2f(0.65, 1.0));
	//drawTex(pEoc->getOptixTex(), true, nv::vec2f(0.0, 1.0-0.8 / ROWLARGER), nv::vec2f(0.8, 1.0));
//	drawTex(pEoc->getOptixTex(), true, nv::vec2f(0.0, 1.0 - 0.8 / ROWLARGER), nv::vec2f(0.8, 1.0));//
	drawTex(pEoc->getOptixTex(), true, nv::vec2f(0.0, 0), nv::vec2f(0.75, 0.75));//

	drawTex(pEoc->getRightEocBuffer()->getTexture(0), true, nv::vec2f(0.75, 0.00), nv::vec2f(1, 0.25));
	drawTex(pEoc->getTopEocBuffer()->getTexture(0), true, nv::vec2f(0.75, 0.25), nv::vec2f(1, 0.5));
	//drawTex(pEoc->getTopOccludeFbo()->getTexture(0), true, nv::vec2f(0.75, 0.50), nv::vec2f(1, 0.75));
	drawTex(pEoc->getGbufferP()->getTexture(0), true, nv::vec2f(0.75, 0.75));
	g_Consturctor.construct();
	drawTex(g_Consturctor.getReconstructTexture(), true, nv::vec2f(0., 0.6), nv::vec2f(0.4, 1.0));
	g_Consturctor.render(g_bufferShader, texManager);// 这是做对比的方法
	CHECK_ERRORS();
	
	

	drawTex(g_Consturctor.getBuffer().getTexture(0), true, nv::vec2f(0.75, 0.5), nv::vec2f(1, 0.75));

	//drawTex(pEoc->getRenderFbo()->getTexture(0), true, nv::vec2f(0.0, 0.0), nv::vec2f(0.75, 0.50));

	if (drawFps ) {
		static char fps_text[32];
		float fps = g_time.getFps();
		sprintf(fps_text, "fps: %6.1f", fps);
		drawText(fps_text, 30.0f, 80.0f, GLUT_BITMAP_8_BY_13);
	}
	glutSwapBuffers();
	glutReportErrors();
}
void idle()
{
	if (!_isNaviCam)
	{
		g_Camera.Update();
	}
	else
	{
		g_navi_Cam.Update();
	}
	glutPostRedisplay();
	
	
}

int main(int argc, char** argv)
{
	freopen("stdout.txt", "w", stdout);

	freopen("stderr.txt", "w", stderr);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
	glutCreateWindow("框架");
	Init();
	glutDisplayFunc(Display);
	glutReshapeFunc(Reshape);
	glutKeyboardFunc(key);
	glutIdleFunc(idle);
	glutMainLoop();
	return 0;
}



#include<stdlib.h>
#include<GL/glew.h>

#include<fstream>
#include"nvMath.h"
//#include<glm/vec2.hpp>

#define CHECK_ERRORS()         \
	do {                         \
	GLenum err = glGetError(); \
	if (err) {                                                       \
	printf( "GL Error %d at line %d of FILE %s\n", (int)err, __LINE__,__FILE__);       \
	exit(-1);                                                      \
	}                                                                \
	} while(0)
#ifndef TEXSTATE_H
#define TEXSTATE_H

class TexState{
public:
	TexState(){};

	TexState(int w, int h, int f = GL_LINEAR, int c = GL_CLAMP) :width(w), height(h), filterMode(f), clampMode(c){

	};
	inline int getWidth(){ return width; };
	inline int getHeight(){ return height; };
	inline int getClamp(){ return clampMode; };
	inline int getFilterMode(){ return filterMode; };

private:
	int width, height;
	int filterMode, clampMode;
};
#endif

#ifndef FBO_H
#define FBO_H
static GLenum mybuffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT,
                           GL_COLOR_ATTACHMENT4_EXT, GL_COLOR_ATTACHMENT5_EXT, GL_COLOR_ATTACHMENT6_EXT };
class Fbo{
private:

	GLuint DepthTex;
	GLuint depthbuffer;
	int beforeFboId;
	TexState texDescript;
	       //�����������

public:
	int num;    
	//�洢����
	GLuint TexId[8];
	GLuint fboId;   // ���fbo��ID

	
	inline GLuint getFboId()
	{
		return fboId;
	}
	inline bool isScreen()
	{
		return fboId == 0;
	}
	Fbo(){beforeFboId = 0;fboId = 0;};
	Fbo(int num, int width, int height);
	void set(int num, int width,int height);
	~Fbo();
	inline TexState getTexState(){
		return texDescript;
	}
	//��ʼ�� �����ڴ�
	void init( GLenum MAGLINER = GL_NEAREST,GLenum MINLINER = GL_NEAREST);
	//����fbo
	void attachId();
	void begin(nv::vec3f clearColor = nv::vec3f(0,0,0),bool clear = true);

	//ָ����Ⱦbuffer
	void BindForWrite(int i);
	void BindForRead(int i, GLenum TextureUnit);
	void copyFrom(Fbo & source,int i =0);
	void copyFromBuffer(Fbo & source,int i=0);
	//ͣ��fbo

	void end();
	//static void SaveBMP(char *filename);
	void destory();
	//��������ָ��
	GLuint getTexture(int id);
	//��̬���� �洢bmp
	static void SaveBMP(const char *fileName, unsigned char   *buf, unsigned int width, unsigned int height);
	static void SaveBMP(GLuint texId, const char * fileName, unsigned int width, unsigned int height);
	nv::vec4f debugPixel(int id, int x,int y,int scale = 1);
	void SaveBMP(const char *fileName, int id);

	void SaveBuffToBMP(const char *fileName, int id);

	void SaveFloat(const char *fileName, int id);

	void CoutFloat(int);

	int ComputNum(int id, int lastNum);
	static void drawScreenBackBuffer(int w, int h,int fobId = 0);
	static void saveScreen(std::string fileName,int width,int height);
	static void mipMaptoSceen(GLuint texid,int widht,int height);
	//glm::vec2 ComputError(int);
};
#endif
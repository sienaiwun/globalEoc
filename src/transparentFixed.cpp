#include "transparentFixed.h"
#include <algorithm>

#define CHECK_ERRORS()         \
	do {                         \
	GLenum err = glGetError(); \
	if (err) {                                                       \
	printf( "GL Error %d at line %d of FILE %s\n", (int)err, __LINE__,__FILE__);       \
	exit(-1);                                                      \
			}                                                                \
			} while(0)


static const char* readCSShaderFile(const char* shaderFileName)
{
	FILE* fp = fopen(shaderFileName, "r");

	if (fp == NULL) { return NULL; }

	fseek(fp, 0L, SEEK_END);
	long size = ftell(fp);

	fseek(fp, 0L, SEEK_SET);
	char* buf = new char[size + 1];
	fread(buf, 1, size, fp);

	buf[size] = '\0';
	fclose(fp);

	return buf;
}


static void genComputeProg(GLuint &csPro, const char *cs_shaderFile)
{
	// Creating the compute shader, and the program object containing the shader
	csPro = glCreateProgram();
	GLuint cs = glCreateShader(GL_COMPUTE_SHADER);

	const char *csSrc = readCSShaderFile(cs_shaderFile);
	if (csSrc == NULL)
	{
		printf("failed to read %s\n", cs_shaderFile);
		exit(EXIT_FAILURE);
	}

	// In order to write to a texture, we have to introduce it as image2D.
	// local_size_x/y/z layout variables define the work group size.
	// gl_GlobalInvocationID is a uvec3 variable giving the global ID of the thread,
	// gl_LocalInvocationID is the local index within the work group, and
	// gl_WorkGroupID is the work group's index
	/*const char *csSrc[] = {
	"#version 430\n",
	"uniform float roll;\
	uniform image2D destTex;\
	layout (local_size_x = 16, local_size_y = 16) in;\
	void main() {\
	ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);\
	float localCoef = length(vec2(ivec2(gl_LocalInvocationID.xy)-8)/8.0);\
	float globalCoef = sin(float(gl_WorkGroupID.x+gl_WorkGroupID.y)*0.1 + roll)*0.5;\
	imageStore(destTex, storePos, vec4(1.0-globalCoef*localCoef, 0.0, 0.0, 0.0));\
	}"
	};*/

	glShaderSource(cs, 1, (const GLchar**)&csSrc, NULL);
	glCompileShader(cs);
	int rvalue;
	glGetShaderiv(cs, GL_COMPILE_STATUS, &rvalue);
	if (!rvalue) {
		fprintf(stderr, "Error in compiling the compute shader\n");
		GLchar log[10240];
		GLsizei length;
		glGetShaderInfoLog(cs, 10239, &length, log);
		fprintf(stderr, "Compiler log:\n%s\n", log);
		exit(40);
	}
	glAttachShader(csPro, cs);

	glLinkProgram(csPro);
	glGetProgramiv(csPro, GL_LINK_STATUS, &rvalue);
	if (!rvalue) {
		fprintf(stderr, "Error in linking compute shader program\n");
		GLchar log[10240];
		GLsizei length;
		glGetProgramInfoLog(csPro, 10239, &length, log);
		fprintf(stderr, "Linker log:\n%s\n", log);
		exit(41);
	}

}

static GLuint getAtomicCounter(GLuint buffer)
{
	// read back total fragments that been shaded
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, buffer);
	GLuint *ptr = (GLuint *)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), GL_MAP_READ_BIT);
	GLuint fragsCount = *ptr;
	glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
	return fragsCount;
}
#include "gbuffershader.h"
OITFixedRender::OITFixedRender(int w, int h, int k) :m_height(h), m_width(w), m_k(k), m_pScene(NULL)
{
	m_renderFbo = Fbo(1, m_width, m_height);
	m_renderFbo.init();
	/*
	std::vector<nv::vec4f> testImage;
	testImage.resize(m_width*m_height);
	auto func = [](nv::vec4f& source){source = nv::vec4f(1, 0, 0, 1); };
	std::for_each(testImage.begin(), testImage.end(), func);

	CHECK_ERRORS();
	glBindTexture(GL_TEXTURE_2D, m_renderFbo.getTexture(0));

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, m_width, m_height, 0, GL_RGBA, GL_FLOAT, testImage.data());


	CHECK_ERRORS();
	*/
	m_total_pixel = m_width*m_height;

	glGenTextures(1, &m_head_pointer_texture);
	glBindTexture(GL_TEXTURE_2D, m_head_pointer_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, m_width, m_height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);


	glGenBuffers(1, &m_head_pointer_initializer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_head_pointer_initializer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, m_total_pixel * sizeof(GLuint), NULL, GL_STATIC_DRAW);

	m_data = (GLuint*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
	memset(m_data, 0x00, m_total_pixel * sizeof(GLuint));
	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glGenBuffers(1, &m_atomic_counter_array_buffer);
	glBindBuffer(GL_TEXTURE_BUFFER, m_atomic_counter_array_buffer);
	glBufferData(GL_TEXTURE_BUFFER, m_total_pixel*sizeof(GLuint), NULL, GL_DYNAMIC_COPY);

	glGenTextures(1, &m_atomic_counter_array_buffer_texture);
	glBindTexture(GL_TEXTURE_2D, m_atomic_counter_array_buffer_texture);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, m_fragment_storage_buffer);

	glGenBuffers(1, &m_atomic_counter_buffer);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_atomic_counter_buffer);
	glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), NULL, GL_DYNAMIC_COPY);
	glBindTexture(GL_TEXTURE_BUFFER, 0);

	glGenBuffers(1, &m_fragment_storage_buffer);
	glBindBuffer(GL_TEXTURE_BUFFER, m_fragment_storage_buffer);
	glBufferData(GL_TEXTURE_BUFFER, m_k*m_total_pixel*sizeof(GLfloat) * 4, NULL, GL_DYNAMIC_COPY);



	glGenTextures(1, &m_linked_list_texture);
	glBindTexture(GL_TEXTURE_BUFFER, m_linked_list_texture);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32UI, m_fragment_storage_buffer);
	glBindTexture(GL_TEXTURE_BUFFER, 0);
	glBindImageTexture(1, m_linked_list_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32UI);

	glBindTexture(GL_TEXTURE_2D, 0);
	CHECK_ERRORS();


	m_oitShader.init();

	//glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, m_width, m_height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);


	// computer shader

	CHECK_ERRORS();

	char str[4096];
	GLint shadersLinked = GL_FALSE;
	static const struct
	{
		GLuint num_groups_x;
		GLuint num_groups_y;
		GLuint num_groups_z;
	} dispatch_params = { m_width / 16, m_height / 16, 1 };

	// RENDER SHADER
	genComputeProg(m_computerShader, "./shader/render_fixed_list.glsl");
	CHECK_ERRORS();
	glUseProgram(m_computerShader);

	glGenBuffers(1, &dispatch_buffer);
	glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, dispatch_buffer);

	glBufferData(GL_DISPATCH_INDIRECT_BUFFER, sizeof(dispatch_params), &dispatch_params, GL_STATIC_DRAW);
	glUseProgram(0);
}
void OITFixedRender::render(Camera * pCamera, textureManager & manager)
{

	assert(m_pScene != NULL); 
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_head_pointer_initializer);

	glBindTexture(GL_TEXTURE_2D, m_head_pointer_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, m_width, m_height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);
	glBindTexture(GL_TEXTURE_2D, m_atomic_counter_array_buffer_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, m_width, m_height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);

	glBindImageTexture(0, m_head_pointer_texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
	glBindImageTexture(1, m_linked_list_texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32UI);
	glBindImageTexture(2, m_atomic_counter_array_buffer_texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
	glBindImageTexture(3, m_renderFbo.getTexture(0), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
	
	m_oitShader.begin();
	m_pScene->render(m_oitShader, manager, pCamera);
	m_oitShader.end();



	glBindImageTexture(0, m_head_pointer_texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
	glBindImageTexture(1, m_linked_list_texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32UI);
	glBindImageTexture(2, m_atomic_counter_array_buffer_texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
	glBindImageTexture(3, m_renderFbo.getTexture(0), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);



	
	glUseProgram(m_computerShader);
	glDispatchComputeIndirect(0);
	glUseProgram(0);
	
	return;
	extern GbufferShader g_bufferShader;
	m_renderFbo.begin();
	CHECK_ERRORS();
	m_pScene->render(m_oitShader, manager, pCamera);
	CHECK_ERRORS();
	m_renderFbo.end();
	CHECK_ERRORS();



}
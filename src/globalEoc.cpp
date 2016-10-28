#include "globaleoc.h"
#include <algorithm>
#include "myGeometry.h"
#include "quad.h"

#define GL_CONSERVATIVE_RASTERIZATION_NV 0x9346
#define CHECK_ERRORS()         \
	do {                         \
	GLenum err = glGetError(); \
	if (err) {                                                       \
	printf( "GL Error %d at line %d of FILE %s\n", (int)err, __LINE__,__FILE__);       \
	exit(-1);                                                      \
			}                                                                \
			} while(0)

EOCrender::EOCrender()
{
	m_eocRightCam = EocCamera(is_Right, 5, 50);
	m_debugSwap = false;
}
EOCrender::EOCrender(int w, int h) :m_height(h), m_width(w), m_pScene(NULL)
{
	m_renderFbo = Fbo(1, m_width, m_height);
	m_renderFbo.init();

	m_gbufferFbo = Fbo(3, m_width, m_height);
	m_gbufferFbo.init();
	m_edgeFbo = Fbo(1, m_width, m_height);
	m_edgeFbo.init();
	m_progFbo = Fbo(1, m_width, m_height);
	m_progFbo.init();

	m_gbufferEocFbo = Fbo(1, m_width, m_height);
	m_gbufferEocFbo.init();
	glBindTexture(GL_TEXTURE_2D, m_edgeFbo.getTexture(0));
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindTexture(GL_TEXTURE_2D, m_progFbo.getTexture(0));
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	m_occludedBuffer = Fbo(1, m_width, m_height);
	m_occludedBuffer.init();

	m_edgeShader.init();
	m_edgeShader.setGbuffer(&m_gbufferFbo);
	m_edgeShader.setRes(nv::vec2f(m_width, m_height));


	m_gbufferShader.init();

	m_volumnShader.init();
	m_volumnShader.setRes(nv::vec2f(m_width, m_height));



	g_progShader.init();
	g_progShader.setGbuffer(&m_gbufferFbo);
	g_progShader.setRes(nv::vec2f(m_width, m_height));
	g_progShader.setEdgeFbo(&m_edgeFbo);

	m_blendShader.init();

	//m_pQuad = new QuadScene();
	m_eocRightCam = EocCamera(is_Right, 5, 50);
	m_debugSwap = false;

	pCounter = new RowCounter(w, h);
	pCounter->setGbuffer(&m_gbufferFbo);
	pCounter->setOccludorBuffer(&m_occludedBuffer);

	pCounter->setEdgeBuffer(&m_edgeFbo);
	pCounter->init();

	myGeometry::initImageMesh(m_width, m_height);
}
#ifdef OPTIX
void EOCrender::initOptix()
{
	int cudaTexWidth = ROWLARGER * m_width;
	glGenBuffers(1, &m_optixPbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_optixPbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, cudaTexWidth * m_height * sizeof(float4), 0, GL_STREAM_READ);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glGenTextures(1, &m_optixTex);
	glBindTexture(GL_TEXTURE_2D, m_optixTex);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, cudaTexWidth, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	const int optixWidth = cudaTexWidth, optixHeight = m_height;
	try
	{
		m_rtContext = optix::Context::create();
		m_rtContext->setRayTypeCount(1);
		m_rtContext->setEntryPointCount(1);
		m_rtContext["shadow_ray_type"]->setUint(0u);
		m_rtContext["scene_epsilon"]->setFloat(1e-4f);
		m_rtContext["resolution"]->setFloat(cudaTexWidth, m_height);

		std::vector<int> enabled_devices = m_rtContext->getEnabledDevices();
		m_rtContext->setDevices(enabled_devices.begin(), enabled_devices.begin() + 1);
		
		m_rtTexture = m_rtContext->createTextureSamplerFromGLImage(getCudaTex(), RT_TARGET_GL_TEXTURE_2D);
		m_rtTexture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
		m_rtTexture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
		m_rtTexture->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
		m_rtTexture->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
		m_rtTexture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
		m_rtTexture->setMaxAnisotropy(1.0f);
		m_rtTexture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
		m_rtContext["request_texture"]->setTextureSampler(m_rtTexture);

		m_rtfinalBuffer = m_rtContext->createBufferFromGLBO(RT_BUFFER_OUTPUT, m_optixPbo);
		m_rtfinalBuffer->setSize(cudaTexWidth, m_height);
		m_rtfinalBuffer->setFormat(RT_FORMAT_FLOAT4);
		m_rtContext["result_buffer"]->setBuffer(m_rtfinalBuffer);

		
		m_rtContext->setRayGenerationProgram(0, m_rtContext->createProgramFromPTXFile(RAYTRACINGPATH, "shadow_request"));
		m_rtContext->setExceptionProgram(0, m_rtContext->createProgramFromPTXFile(RAYTRACINGPATH, "exception"));
		
		myGeometry::p_rtContext = &m_rtContext;
		m_pScene->setOptix(&m_rtContext);
		m_pScene->optixInit();
		m_rtContext->setStackSize(2048);
		m_rtContext->validate();
	}
	catch (optix::Exception& e)
	{
		printf("%s\n", e.getErrorString().c_str());
		exit(-1);
	}
}
void EOCrender::optixTracing()
{
	int cudaTexWidth = ROWLARGER * m_width;
	try {

		nv::matrix4f modelView = nv::matrix4f(pOriginCam->getModelViewMat());
		nv::matrix4f modelViewInv = inverse(modelView);
		m_rtContext["optixModeView_Inv"]->setMatrix4x4fv(false, modelViewInv.get_value());
		
		//nv::matrix4f modeViewInv = nv::matrix4f(pOriginCam->getModelViewInvMat());
		m_rtContext["eye_pos"]->setFloat(pOriginCam->getCameraPos().x, pOriginCam->getCameraPos().y, pOriginCam->getCameraPos().z);
		Camera * pEoc_camera = m_eocRightCam.getEocCameraP();
		m_rtContext["eoc_eye_pos"]->setFloat(pEoc_camera->getCameraPos().x, pEoc_camera->getCameraPos().y, pEoc_camera->getCameraPos().z);
	
		m_rtContext["bbmin"]->setFloat(pOriginCam->getImageMin().x, pOriginCam->getImageMin().y);
		m_rtContext["bbmax"]->setFloat(pOriginCam->getImageMax().x, pOriginCam->getImageMax().y);

		m_rtContext->launch(0, cudaTexWidth, m_height);
	}
	catch (optix::Exception& e) {
		printf("%s\n", e.getErrorString().c_str());
		exit(1);
	}
	glPushAttrib(GL_PIXEL_MODE_BIT);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_optixPbo);
	glBindTexture(GL_TEXTURE_2D, m_optixTex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cudaTexWidth, m_height,
		GL_RGBA, GL_FLOAT, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glPopAttrib();

}
#endif
void EOCrender::render(textureManager & manager)
{
	static bool once = true;;
	if (once)
	{
		m_eocRightCam.Look();
		//once = false;
	}
	Camera * pRenderCamera;
	if (m_debugSwap == false)
		pRenderCamera = pOriginCam;
	else
		pRenderCamera = m_eocRightCam.getEocCameraP();

	assert(m_pScene != NULL);
	m_gbufferFbo.begin();
	m_pScene->render(m_gbufferShader, manager, pRenderCamera);
	//m_gbufferFbo.SaveBMP("gbuffer.bmp", 0);
	m_gbufferFbo.end();

	//glCullFace(GL_BACK);
	glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);
	m_edgeShader.setCamera(pRenderCamera);
	m_edgeFbo.begin();
	m_pScene->render(m_edgeShader, manager, pRenderCamera);
	//myGeometry::drawQuad(m_edgeShader);
	//m_edgeFbo.SaveBMP("edge.bmp", 0);
	//m_edgeFbo.debugPixel(0, 719, 533);
	//m_edgeFbo.debugPixel(0,340, 807);
	m_edgeFbo.end();
	glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);

	m_progFbo.begin();
	myGeometry::drawQuad(g_progShader);
	m_progFbo.end();
	m_volumnShader.setCamera(pOriginCam);
	m_volumnShader.setGbuffer(&m_gbufferFbo);
	m_volumnShader.setDiffuse(nv::vec3f(1, 0, 0));
	m_volumnShader.setEdgeFbo(&m_progFbo);
	CHECK_ERRORS();
	m_volumnShader.setEocCamera(m_eocRightCam.getEocCameraP()->getCameraPos());
	CHECK_ERRORS();

	m_occludedBuffer.begin();
	CHECK_ERRORS();
	myGeometry::drawQuadMesh(m_volumnShader, m_width, m_height);
	//m_pScene->render(m_volumnShader, manager, pRenderCamera);
	//m_occludedBuffer.SaveBMP("test.bmp", 0);
	//m_occludedBuffer.debugPixel(0, 512, 512);
	CHECK_ERRORS();

	m_occludedBuffer.end();
	CHECK_ERRORS();


	/*
	m_blendShader.setBuffer1(&m_gbufferFbo);
	m_blendShader.setBuffer2(&m_edgeFbo);

	m_renderFbo.begin();
	myGeometry::drawQuad(m_blendShader);
	//m_renderFbo.SaveBMP("test2.bmp", 0);

	m_renderFbo.end();
	*/
	m_gbufferEocFbo.begin();
	m_pScene->render(m_gbufferShader, manager, m_eocRightCam.getEocCameraP());
	m_gbufferEocFbo.end();

	pCounter->render(pOriginCam, m_eocRightCam.getEocCameraP());
	/*
	nv::vec3f cameraCamPos = pOriginCam->getCameraPos();
	nv::vec3f eocCameraPos = m_eocRightCam.getEocCameraP()->getCameraPos();
	nv::vec3f leftEdgePos = nv::vec3f(-4.312300, -12.039669, -42.240936);
	nv::vec3f worldPos = nv::vec3f(-4.312305, -12.491903, -51.640202);
	nv::vec3f line1 = normalize(worldPos - eocCameraPos);
		
	nv::vec3f line2 = normalize(leftEdgePos - cameraCamPos);
	nv::vec3f zhijiao = normalize(cross(line1, line2));
	nv::vec3f cuixian = normalize(cross(line2, zhijiao));
	float dis = (dot(cameraCamPos - eocCameraPos, cuixian) / (dot(line1, cuixian)));
	nv::vec3f tangPos = eocCameraPos + line1 * dis;
	nv::vec3f line3 = tangPos - cameraCamPos;
	float dotV = dot(line3, zhijiao);
	*/
#ifdef OPTIX
	optixTracing();
#endif

}
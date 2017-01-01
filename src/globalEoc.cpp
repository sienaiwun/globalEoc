#include "globaleoc.h"
#include <algorithm>
#include "myGeometry.h"
#include "quad.h"
#include "macro.h"

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
	m_eocRightCam = EocCamera(is_Right, dis_orgin, to_flocus);
	m_eocTopCam = EocCamera(is_Top, dis_orgin, to_flocus);
	m_debugSwap = false;
	pOriginCam = new Camera();
}
EOCrender::EOCrender(int w, int h) :m_height(h), m_width(w), m_pScene(NULL)
{
	pOriginCam = new Camera();
	m_renderFbo = Fbo(1, m_width, m_height);
	m_renderFbo.init();

	m_gbufferFbo = Fbo(3, m_width, m_height);
	m_gbufferFbo.init();
	m_edgeFbo = Fbo(1, m_width, m_height);
	m_edgeFbo.init();
	m_progFbo = Fbo(2, m_width, m_height);
	m_progFbo.init();

	m_gbufferRightEocFbo = Fbo(1, m_width, m_height);
	m_gbufferRightEocFbo.init();

	m_gbufferTopEocFbo = Fbo(1, m_width, m_height);
	m_gbufferTopEocFbo.init();

	glBindTexture(GL_TEXTURE_2D, m_edgeFbo.getTexture(0));
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindTexture(GL_TEXTURE_2D, m_progFbo.getTexture(0));
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindTexture(GL_TEXTURE_2D, m_progFbo.getTexture(1));
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);


	m_occludedRightBuffer = Fbo(1, m_width, m_height);
	m_occludedRightBuffer.init();


	m_occludedTopBuffer = Fbo(1, m_width, m_height);
	m_occludedTopBuffer.init();

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
	m_eocRightCam = EocCamera(is_Right, dis_orgin, to_flocus);
	//m_eocTopCam = EocCamera(is_Top, dis_orgin, to_flocus);
	m_eocTopCam = EocCamera(is_Top, 0, to_flocus);
	m_debugSwap = false;

	pCounter = new RowCounter(w, h);
	pCounter->setGbuffer(&m_gbufferFbo);
	pCounter->setRightOccludorBuffer(&m_occludedRightBuffer);
	pCounter->setTopOccludorBuffer(&m_occludedTopBuffer);

	pCounter->setEdgeBuffer(&m_edgeFbo);
	pCounter->init();

	myGeometry::initImageMesh(m_width, m_height);
}
#ifdef OPTIX
void EOCrender::initOptix()
{
	m_cudaTexWidth = ROWLARGER * m_width;
	m_cudaTexHeight = ROWLARGER * m_height;
	glGenBuffers(1, &m_optixPbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_optixPbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, m_cudaTexWidth * m_cudaTexHeight * sizeof(float4), 0, GL_STREAM_READ);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glGenTextures(1, &m_optixTex);
	glBindTexture(GL_TEXTURE_2D, m_optixTex);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, m_cudaTexWidth ,m_cudaTexHeight, 0, GL_RGBA, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);


	
	glGenBuffers(1, &m_optixWorldPbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_optixWorldPbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, m_cudaTexWidth * m_cudaTexHeight  * sizeof(float4), 0, GL_STREAM_READ);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glGenTextures(1, &m_optixWorldTex);
	glBindTexture(GL_TEXTURE_2D, m_optixWorldTex);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, m_cudaTexWidth, m_cudaTexHeight, 0, GL_RGBA, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	const int optixWidth = m_cudaTexWidth, optixHeight = m_height;
	try
	{
		m_rtContext = optix::Context::create();
		m_rtContext->setRayTypeCount(1);
		m_rtContext->setEntryPointCount(1);
		m_rtContext["shadow_ray_type"]->setUint(0u);
		m_rtContext["scene_epsilon"]->setFloat(1e-4f);
		m_rtContext["resolution"]->setFloat(m_cudaTexWidth, m_cudaTexHeight);

		std::vector<int> enabled_devices = m_rtContext->getEnabledDevices();
		m_rtContext->setDevices(enabled_devices.begin(), enabled_devices.begin() + 1);
		
		m_rtTexture = m_rtContext->createTextureSamplerFromGLImage(getCudaTopTex(), RT_TARGET_GL_TEXTURE_2D);
		m_rtTexture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
		m_rtTexture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
		m_rtTexture->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
		m_rtTexture->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
		m_rtTexture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
		m_rtTexture->setMaxAnisotropy(1.0f);
		m_rtTexture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
		m_rtContext["request_texture"]->setTextureSampler(m_rtTexture);

		m_rtfinalBuffer = m_rtContext->createBufferFromGLBO(RT_BUFFER_OUTPUT, m_optixPbo);
		m_rtfinalBuffer->setSize(m_cudaTexWidth, m_cudaTexHeight);
		m_rtfinalBuffer->setFormat(RT_FORMAT_FLOAT4);
		m_rtContext["result_buffer"]->setBuffer(m_rtfinalBuffer);

		
		m_rtWorldBuffer = m_rtContext->createBufferFromGLBO(RT_BUFFER_OUTPUT, m_optixWorldPbo);
		m_rtWorldBuffer->setSize(m_cudaTexWidth, m_cudaTexHeight);
		m_rtWorldBuffer->setFormat(RT_FORMAT_FLOAT4);
		m_rtContext["position_buffer"]->setBuffer(m_rtWorldBuffer);

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
	int cudaTexHeight = ROWLARGER* m_height;
	try {

		nv::matrix4f modelView = nv::matrix4f(pOriginCam->getModelViewMat());
		nv::matrix4f modelViewInv = inverse(modelView);
		m_rtContext["optixModeView_Inv"]->setMatrix4x4fv(false, modelViewInv.get_value());
		
		//nv::matrix4f modeViewInv = nv::matrix4f(pOriginCam->getModelViewInvMat());
		m_rtContext["eye_pos"]->setFloat(pOriginCam->getCameraPos().x, pOriginCam->getCameraPos().y, pOriginCam->getCameraPos().z);
		Camera * p_right_Eoc_camera = m_eocRightCam.getEocCameraP();
		Camera * pTopEoc_camera = m_eocTopCam.getEocCameraP();
		m_rtContext["eoc_eye_right_pos"]->setFloat(p_right_Eoc_camera->getCameraPos().x, p_right_Eoc_camera->getCameraPos().y, p_right_Eoc_camera->getCameraPos().z);
		m_rtContext["eoc_eye_top_pos"]->setFloat(pTopEoc_camera->getCameraPos().x, pTopEoc_camera->getCameraPos().y, pTopEoc_camera->getCameraPos().z);

		m_rtContext["bbmin"]->setFloat(pOriginCam->getImageMin().x, pOriginCam->getImageMin().y);
		m_rtContext["bbmax"]->setFloat(pOriginCam->getImageMax().x, pOriginCam->getImageMax().y);
		m_rtContext["optixModelView"]->setMatrix4x4fv(false, modelView.get_value());
		m_rtContext["rightND"]->setFloat(getRightND().x, getRightND().y, getRightND().z);
		m_rtContext["topND"]->setFloat(getTopD().x, getTopD().y, getTopD().z);
		m_rtContext->launch(0, cudaTexWidth, cudaTexHeight);
	}
	catch (optix::Exception& e) {
		printf("%s\n", e.getErrorString().c_str());
		exit(1);
	}
	glPushAttrib(GL_PIXEL_MODE_BIT);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_optixPbo);
	glBindTexture(GL_TEXTURE_2D, m_optixTex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cudaTexWidth, cudaTexHeight,
		GL_RGBA, GL_FLOAT, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glPopAttrib();

	glPushAttrib(GL_PIXEL_MODE_BIT);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_optixWorldPbo);
	glBindTexture(GL_TEXTURE_2D, m_optixWorldTex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cudaTexWidth, cudaTexHeight,
		GL_RGBA, GL_FLOAT, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glPopAttrib();
	
	/*
	glEnable(GL_TEXTURE_2D);
	BYTE *pTexture = NULL;
	pTexture = new BYTE[cudaTexWidth*cudaTexHeight * 3];
	memset(pTexture, 0, cudaTexWidth*cudaTexHeight * 3 * sizeof(BYTE));

	glBindTexture(GL_TEXTURE_2D, m_optixTex);//TexPosId   PboTex

	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pTexture);

	int w = cudaTexWidth;
	int h = cudaTexHeight;
	Fbo::SaveBMP("optix.bmp", pTexture, w, h);
	if (pTexture)
	delete[] pTexture;
	glBindTexture(GL_TEXTURE_2D, 0);//TexPosId   PboTex
	*/

}
nv::vec2f toScreen(nv::vec4f value, nv::matrix4f mvp)
{
	nv::vec4f worldPos = nv::vec4f(value.y, value.z, value.w, 1);
	nv::vec4f temp = mvp*worldPos;
	temp /= temp.w;
	nv::vec2f xy = nv::vec2f(temp.x, temp.y);
	xy = xy*0.5 + nv::vec2f(0.5, 0.5);
	return xy * 1024;
}
#endif
void EOCrender::render(textureManager & manager)
{
	static bool once = true;;
	if (once)
	{
		m_eocRightCam.Look();
		m_eocTopCam.Look();
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

	//progBuffer里面texture0 存储的是
	m_progFbo.begin();
	myGeometry::drawQuad(g_progShader);
	//m_progFbo.SaveBMP("prog2.bmp", 1);

	//m_progFbo.debugPixel(1, 366, 664);
	//m_progFbo.debugPixel(1, 366, 666);
	//m_progFbo.debugPixel(1, 366, 665);
	//nv::vec2f uv = toScreen(m_progFbo.debugPixel(1, 442, 719), pOriginCam->getMvpMat());
	m_progFbo.end();

	// 设置volumn right
	m_volumnShader.setCamera(pOriginCam);
	m_volumnShader.setGbuffer(&m_gbufferFbo);
	m_volumnShader.setDiffuse(nv::vec3f(1, 0, 0));
	m_volumnShader.setEdgeFbo(&m_progFbo);
	m_volumnShader.setIsVertival(false);
	CHECK_ERRORS();
	m_volumnShader.setEocCamera(m_eocRightCam.getEocCameraP()->getCameraPos());
	CHECK_ERRORS();

	m_occludedRightBuffer.begin();
	CHECK_ERRORS();
	myGeometry::drawQuadMesh(m_volumnShader, m_width, m_height);
	CHECK_ERRORS();
	m_occludedRightBuffer.end();
	CHECK_ERRORS();

	// 设置volumn top
	CHECK_ERRORS();
	m_volumnShader.setEocCamera(m_eocTopCam.getEocCameraP()->getCameraPos());
	m_volumnShader.setIsVertival(true);
	m_volumnShader.setAssoTex(m_occludedRightBuffer.getTexture(0));
	CHECK_ERRORS();

	glDisable(GL_CULL_FACE);
	m_occludedTopBuffer.begin();
	CHECK_ERRORS();
	myGeometry::drawQuadMesh(m_volumnShader, m_width, m_height);
	CHECK_ERRORS();
	//m_occludedTopBuffer.debugPixel(0, 512, 512);
	//m_occludedTopBuffer.SaveBMP("topOccluder.bmp", 0);
	m_occludedTopBuffer.end();
	CHECK_ERRORS();


	
	m_blendShader.setBuffer1(&m_gbufferFbo);
	m_blendShader.setBuffer2(&m_occludedTopBuffer);
	m_renderFbo.begin();
	myGeometry::drawQuad(m_blendShader);
	//m_renderFbo.SaveBMP("temp.bmp", 0);
	//m_renderFbo.SaveBMP("test2.bmp", 0);
	m_renderFbo.end();
	
	//for visualization
	m_gbufferRightEocFbo.begin();
	m_pScene->render(m_gbufferShader, manager, m_eocRightCam.getEocCameraP());
	m_gbufferRightEocFbo.end();

	m_gbufferTopEocFbo.begin();
	m_pScene->render(m_gbufferShader, manager, m_eocTopCam.getEocCameraP());

	m_gbufferTopEocFbo.end();

	pCounter->render(pOriginCam, m_eocRightCam.getEocCameraP(),m_eocTopCam.getEocCameraP());
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
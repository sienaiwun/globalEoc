#include "Fbo.h"
#include "cuda.h"


void CudaTexResourse::init()
{
	if (m_texture)
		checkCudaErrors(cudaGraphicsGLRegisterImage(&m_CudaReourse, m_texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));

}
void CudaTexResourse::setEveryTex(int texId)
{
	//Fbo::SaveBMP(texId,"./test/passReflection.bmp",1024,1024);
	checkCudaErrors(cudaGraphicsUnregisterResource(m_CudaReourse));
	m_tempTex = texId;
	checkCudaErrors(cudaGraphicsGLRegisterImage(&m_CudaReourse, m_tempTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
}
void CudaTexResourse::set(int texId)
{
	setEveryTex(texId);

}
void CudaTexResourse::map()
{

	checkCudaErrors(cudaGraphicsMapResources(1, &m_CudaReourse, 0));
	cudaRelateTex(this);

}
void CudaTexResourse::unmap()
{
	checkCudaErrors(cudaGraphicsUnmapResources(1, &m_CudaReourse, 0));
}

/////////////////////////// pbo tex ////////////////////////
void CudaPboResource::init()
{
	initPbo();
	initTex();
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_CudaReourse, m_pbo, cudaGraphicsMapFlagsWriteDiscard));

}
void CudaPboResource::initPbo()
{
	glGenBuffers(1, &m_pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
	if (is_texture())
	{
		glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width*m_height*sizeof(float4), 0, GL_STREAM_READ);
	}
	else if (m_type == float2_t)
	{
		glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width*m_height*sizeof(float2), 0, GL_STREAM_READ);

	}
	else if (m_type == list_e || m_type == list_top_e)
	{
		glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width*m_height*sizeof(ListNote), 0, GL_STREAM_READ);
	}

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

}
void CudaPboResource::initTex()
{
	if (is_texture())
	{
		glGenTextures(1, &m_texture);
		glBindTexture(GL_TEXTURE_2D, m_texture);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
}
void CudaPboResource::generateTex()
{
	if (m_type == list_e||m_type == list_top_e)
	{
		// 把list_note不需要转换成纹理

#ifdef DEBUG
		glPushAttrib(GL_PIXEL_MODE_BIT);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
		glBindTexture(GL_TEXTURE_2D, m_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_INT, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glPopAttrib();
		
		glEnable(GL_TEXTURE_2D);
		int *pTexture = NULL;
		pTexture = new int[m_width*m_height * 3];
		memset(pTexture, 0, m_width*m_height * 3* sizeof(int));


		glBindTexture(GL_TEXTURE_2D, m_texture);//TexPosId   PboTex


		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_INT, pTexture);

		for (int y = 0; y < m_height; y++)
		{
		printf("%d index:%d\t", y, pTexture[3 * y]);
		}
		glBindTexture(GL_TEXTURE_2D, 0);//TexPosId   PboTex*/

#endif

	}
	else if (is_texture())
	{


		glPushAttrib(GL_PIXEL_MODE_BIT);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
		glBindTexture(GL_TEXTURE_2D, m_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glPopAttrib();

		
		
	
		if (m_type == construct_t)
		{
			
			glEnable(GL_TEXTURE_2D);
			BYTE *pTexture = NULL;
			pTexture = new BYTE[m_width*m_height * 3];
			memset(pTexture, 0, m_width*m_height * 3 * sizeof(BYTE));


			glBindTexture(GL_TEXTURE_2D, m_texture);//TexPosId   PboTex


			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pTexture);


			Fbo::SaveBMP("b.bmp", pTexture, m_width, m_height);


			glBindTexture(GL_TEXTURE_2D, 0);
			
		}
		

		
	}


}
void CudaPboResource::map()
{
	checkCudaErrors(cudaGraphicsMapResources(1, &m_CudaReourse, 0));
	cudaRelateArray(this);

}
void CudaPboResource::unMap()
{
	checkCudaErrors(cudaGraphicsUnmapResources(1, &m_CudaReourse, 0));
}

#include "cuda.h"
#include <helper_math.h>
#include <nvMatrix.h>
#include "Camera.h"
#define FLT_MAX 99999.9
texture<float4, 2, cudaReadModeElementType> cudaProgTex;//记录的是Edge,edge里面x记录x的sobal,edge里面y记录y的sobal
texture<float4, 2, cudaReadModeElementType> cudaOccuderTex;
texture<float4, 2, cudaReadModeElementType> cudaTopOccuderTex;
texture<float4, 2, cudaReadModeElementType> cudaColorTex;
texture<float4, 2, cudaReadModeElementType> cudaPosTex;

cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

typedef enum {
	isVolumn,
	notVolumn,
}pixelEnum;


uint3 *cuda_PBO_Buffer;

__device__ uint3* d_cudaPboBuffer;
float4 *cuda_TexturePbo_buffer, *cuda_top_TexturePbo_buffer;
__device__ float4* d_cudaTexture;
__device__ float4* d_cudaTopTexture;
__device__ int imageWidth, imageHeight, d_outTextureWidth, d_outTextureHeigh, d_outTopTextureWidth, d_outTopTextureHeight;
__device__ int d_index;
__device__ ListNote* d_listBuffer;
__device__ ListNote* d_listBuffer_top;
__device__ int d_atomic;
__device__ float3 d_cameraPos;
__device__ float3 d_eocPos;
__device__ float3 d_eocTopPos;
float* modelView;
__device__ float* d_modelView;

__device__ void MutiMatrix(float * Matrix, float x, float y, float z, float &outx, float &outy, float &outz)
{
	float tempx = x*Matrix[0] + y*Matrix[1] + z*Matrix[2] + Matrix[3];
	float tempy = x*Matrix[4] + y*Matrix[5] + z*Matrix[6] + Matrix[7];
	float tempz = x*Matrix[8] + y*Matrix[9] + z*Matrix[10] + Matrix[11];
	float tempt = x*Matrix[12] + y*Matrix[13] + z*Matrix[14] + Matrix[15];
	float3 result;
	if (tempt<0.0001)
	{
		outx = FLT_MAX;
		outy = FLT_MAX;
		outz = FLT_MAX;
	}
	outx = tempx / tempt;
	outy = tempy / tempt;
	outz = tempz / tempt;
}


class List
{

};
__device__ float repo(float value)
{
	return 1.0f / value;
}
__device__ float3 projective_interpo(float3 beginPos, float3 endPos, float ratio)
{
	float x, y, z1, z2;
	MutiMatrix(d_modelView, beginPos.x, beginPos.y, beginPos.z, x, y, z1);
	MutiMatrix(d_modelView, endPos.x, endPos.y, endPos.z, x, y, z2);
	float real_z = repo(ratio *repo(z1) + (1 - ratio) * repo(z2));
	float real_ratio = (real_z - z1) / (z2 - z1);
	return beginPos* real_ratio + endPos  * (1 - real_ratio);
}
__device__ bool isVolume(float2 uv, int *state)
{
	//if (uv.x > 230 && uv.x < 300)
	//	return true;
	float4 value = tex2D(cudaOccuderTex, uv.x, uv.y);
	return value.x > 0.5;
}
__device__ bool isVolumeTop(float2 uv, int * state)
{
	float4 value = tex2D(cudaTopOccuderTex, uv.x, uv.y);
	return value.x > 0.5;

}
__device__ bool isEdge(float2 uv, int * state)
{
	return tex2D(cudaProgTex, uv.x, uv.y).x > 0.05;
}
__device__ bool isEdgeTop(float2 uv, int * state)
{
	return tex2D(cudaProgTex, uv.x, uv.y).y > 0.05;
}
__device__ bool isMinusEdge(float2 uv)
{
	return tex2D(cudaProgTex, uv.x, uv.y).x < -0.05;
}
__device__ bool isMinusEdgeTop(float2 uv)
{
	return tex2D(cudaProgTex, uv.x, uv.y).y < -0.05;
}

__device__ float2 toUv(int x, int y)
{
	return make_float2(x + 0.5, y + 0.5);
}
__global__ void countRowKernelTop(int kernelWidth, int kernelHeight)
{
	int index = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (index > kernelWidth)
		return;
	//if (index != 512)
	//	return;
	int arrayNum = index;
	int accumNum = 0;
	int state = 0;
	pixelEnum etype = notVolumn;
	unsigned int* nextPtr = &d_cudaPboBuffer[arrayNum].x;
	int listIndex;
	int lastMinusIndey = 0;
	for (int y = 0; y< imageHeight; y++)
	{
		float2 currentUv = toUv(index, y);
		if (isMinusEdgeTop(currentUv))
		{
			//printf("last y:%d\n", y);
			lastMinusIndey = y;
		}
		if (isVolumeTop(currentUv, &state) && etype == notVolumn)
		{
			//printf("insert :%d\n", y);
			listIndex = atomicAdd(&d_atomic, 1);
			atomicExch(nextPtr, listIndex);// write listIndex to next slot
			d_listBuffer[listIndex].beginIndex = y;
			d_listBuffer[listIndex].endIndex = y;
			d_listBuffer[listIndex].nextPt = 0;
			d_listBuffer[listIndex].leftEdge = lastMinusIndey;
			nextPtr = (unsigned int *)(&(d_listBuffer[listIndex].nextPt));

			etype = isVolumn;
		}
		else if (isVolumeTop(currentUv, &state) && etype == isVolumn)
		{


		}
		else if (etype == isVolumn && isEdgeTop(currentUv, &state))
		{
		//	printf("end :%d\n", y);


			d_listBuffer[listIndex].endIndex = y;
			etype = notVolumn;
		}
	}
}
__global__ void countRowKernel(int kernelWidth, int kernelHeight)
{
	int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (y > kernelHeight)
		return;
	//if (y != 807)
	//	return;
	int arrayNum = y;
	int accumNum = 0;
	int state = 0;
	pixelEnum etype = notVolumn;
	unsigned int* nextPtr = &d_cudaPboBuffer[arrayNum].x;
	int listIndex;
	int lastMinusIndex = 0;
	for (int x = 0; x < imageWidth; x++)
	{

		float2 currentUv = toUv(x, y);
		//if (x == 340)

		if (isMinusEdge(currentUv))
		{
			lastMinusIndex = x;
		}
		if (isVolume(currentUv, &state) && etype == notVolumn)
		{
			//printf("insert :%d\n", x);
			listIndex = atomicAdd(&d_atomic, 1);
			atomicExch(nextPtr, listIndex);// write listIndex to next slot
			d_listBuffer[listIndex].beginIndex = x;
			d_listBuffer[listIndex].endIndex = x;
			d_listBuffer[listIndex].nextPt = 0;
			d_listBuffer[listIndex].leftEdge = lastMinusIndex;
			nextPtr = (unsigned int *)(&(d_listBuffer[listIndex].nextPt));
			etype = isVolumn;

		}
		else if (isVolume(currentUv, &state) && etype == isVolumn)
		{


		}
		else if (etype == isVolumn && isEdge(currentUv, &state))
		{
			//printf("end :%d\n", x);

			d_listBuffer[listIndex].endIndex = x;
			etype = notVolumn;
		}

	}
}
__device__ float  myfmax(float a, float b) {
	return ((a) > (b) ? a : b);
}
__device__ float distance(float3 leftPos, float3 currentPos, float3 eocCamera)
{
	float3 line1 = normalize(currentPos - eocCamera);

	float3 line2 = normalize(leftPos - d_cameraPos);
	float3 zhijiao = normalize(cross(line1, line2));
	float3 cuixian = normalize(cross(line2, zhijiao));
	/*printf("currentPos:(%f,%f,%f)\n", currentPos.x, currentPos.y, currentPos.z);
	printf("line1:(%f,%f,%f)\n", line1.x, line1.y, line1.z);
	printf("line2:(%f,%f,%f)\n", line2.x, line2.y, line2.z);
	printf("zhijiao:(%f,%f,%f)\n", zhijiao.x, zhijiao.y, zhijiao.z);
	printf("cuixian:(%f,%f,%f)\n", cuixian.x, cuixian.y, cuixian.z);
	printf("dot:(%f)\n", dot(line1, cuixian));*/
	float dis = (dot(d_cameraPos - eocCamera, cuixian) / (dot(line1, cuixian)));
	if (dis < 0)
		return 1000.0;
	else
		return myfmax(dis, 1.0);

}
__device__ void FillLine(int x)
{
	for (int y = 0; y < d_outTextureHeigh; y++)
	{
		int index = y*d_outTopTextureWidth + x;
		d_cudaTopTexture[index] = d_cudaTexture[y*d_outTextureWidth + x];
	}
}
__device__ float4 FillPoint(int x, int y)
{
	int index = y*d_outTextureWidth + x;
	return  d_cudaTexture[index];
}
__device__ void FillVolumnTop(int beginY, int endY, int x, int endUv, int leftEdge)
{
	int top = min(endY, d_outTopTextureHeight);
	//printf("volumn begin:%d,end:%d,top:%d\n",beginX,endX,top);
	float3 beforePos = make_float3(tex2D(cudaPosTex, x, endUv - 0.5));

	float3 endPos = make_float3(tex2D(cudaPosTex, x, endUv + 0.5));
	float3 leftEdgePos = make_float3(tex2D(cudaPosTex, x, leftEdge + 1.5));
	//printf("endPos:(%f,%f,%f)\n", endPos.x, endPos.y, endPos.z);
	///printf("beforePos:(%f,%f,%f)\n", beforePos.x, beforePos.y, beforePos.z);
	//printf("leftEdgePos:(%f,%f,%f)\n", leftEdgePos.x, leftEdgePos.y, leftEdgePos.z);
	//printf("camera:(%f,%f,%f)\n", d_cameraPos.x, d_cameraPos.y, d_cameraPos.z);
	float3 ecoCamera = d_eocTopPos;
	//printf("eoc:(%f,%f,%f)\n", ecoCamera.x, ecoCamera.y, ecoCamera.z);

	//for (int i = 0; i < 4; i++)
	//	printf("(%f,%f,%f,%f)\n", d_modelView[4 * i + 0], d_modelView[4 * i + 1], d_modelView[4 * i + 2], d_modelView[4 * i + 3]);
	for (int y = beginY; y < top; y++)
	{
		float ratio = (y * 1.0f - beginY*1.0f) / (top - 1 - beginY);
		float3 realPos = projective_interpo(beforePos, endPos, ratio);
		int index = y*d_outTopTextureWidth + x;
		float dis = distance(leftEdgePos, realPos, ecoCamera);
		d_cudaTopTexture[index] = make_float4(-dis, realPos.x, realPos.y, realPos.z);
	}
}
__device__ void FillSpanTop(int beginY, int endY, int x, float2 beginUv, float2 endUv)
{
	int top = min(endY, d_outTopTextureHeight);
	//printf("fill from %d to %d at line %d", beginY, endY, x);
	//printf("begin(%f,%f),end(%f,%f),d_outTextureWidth:%d\n", beginUv.x, beginUv.y, endUv.x, endUv.y, d_outTextureWidth);
	//printf("endY:%d,d_outTopTextureHeight:%d,top:%d\n", endY, d_outTopTextureHeight, top);

	for (int y = beginY; y < top; y++)
	{
		int index = y*d_outTopTextureWidth + x;
		float uvy = beginUv.y + (endUv.y - beginUv.y)*(y - beginY) / (top - beginY);
		d_cudaTopTexture[index] = FillPoint(beginUv.x - 0.5, uvy - 0.5);//tex2D(cudaColorTex, beginUv.x, uvy);
	}
}
__global__ void renderToTexutreTop(int kernelWidth, int kernelHeight)
{
	const int index = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (index > kernelWidth)
		return;
	if (index > imageWidth)
	{
		FillLine(index);
		return;
	}
//	if (index != 512)
//		return;
	int listIndex = index;
	int rowLength = imageWidth;
	ListNote currentNote = *((ListNote*)&d_cudaPboBuffer[listIndex]);
	int texEnd = 0;
	int texBegin = 0;
	int fillBegin = 0;
	int fillEnd = 0;
	int acuumPixel = 0, span = 0;
	//printf("begin:%d,end%d,index:%d\n", currentNote.beginIndex, currentNote.endIndex, currentNote.nextPt);
	/*while (currentNote.nextPt != 0)
	{
	currentNote = d_listBuffer[currentNote.nextPt];
	rowLength += currentNote.endIndex - currentNote.beginIndex;
	//printf("begin:%d,end%d,index:%d,length:%d\n", currentNote.beginIndex, currentNote.endIndex, currentNote.nextPt,rowLength);
	}*/
	//printf("printf:%d\n", rowLength);
	float factor = imageWidth*1.0 / rowLength;
	currentNote = *((ListNote*)&d_cudaPboBuffer[listIndex]);
	int leftEdgeIndex = 0;
	while (currentNote.nextPt != 0)
	{


		currentNote = d_listBuffer[currentNote.nextPt];
		//printf("current:b:%d,e:%d,n:%d,leftEdge:%d\n", currentNote.beginIndex, currentNote.endIndex, currentNote.nextPt, currentNote.leftEdge);

		texEnd = currentNote.endIndex;
		span = currentNote.endIndex - currentNote.beginIndex;
		leftEdgeIndex = currentNote.leftEdge;
		fillBegin = texBegin + acuumPixel;
		fillEnd = texEnd + acuumPixel;
		FillSpanTop(fillBegin*factor, fillEnd*factor, index, toUv(index, texBegin), toUv(index,texEnd));  //for 循环，左闭右开
		FillVolumnTop((fillEnd)*factor, (fillEnd + span)*factor, index, texEnd, leftEdgeIndex);

		acuumPixel += span;
		texBegin = currentNote.endIndex;
		//printf("texBegin:%d,acuumPixel:%d,n:%d\n", texBegin, acuumPixel);

	}
	fillBegin = texBegin + acuumPixel;
	//printf("final:(%d,%d) u(%f,%f)\n", fillBegin, imageWidth + span, toUv(index, texBegin).y, toUv(index,imageWidth).y);
	FillSpanTop(fillBegin*factor, (imageWidth + acuumPixel)*factor, index, toUv(index,texBegin ), toUv(index, imageWidth - 1));

}
__device__ void FillVolumn(int beginX, int endX, int y, int endUv, int leftEdge)
{
	int top = min(endX, d_outTextureWidth);
	//printf("volumn begin:%d,end:%d,top:%d\n",beginX,endX,top);
	float3 beforePos = make_float3(tex2D(cudaPosTex, endUv - 0.5, y));

	float3 endPos = make_float3(tex2D(cudaPosTex, endUv + 0.5, y));
	float3 leftEdgePos = make_float3(tex2D(cudaPosTex, leftEdge + 1.5, y));
	float3 eoc_pos = d_eocPos;
	/*printf("endPos:(%f,%f,%f)\n", endPos.x, endPos.y, endPos.z);
	printf("beforePos:(%f,%f,%f)\n", beforePos.x, beforePos.y, beforePos.z);
	printf("leftEdgePos:(%f,%f,%f)\n", leftEdgePos.x, leftEdgePos.y, leftEdgePos.z);
	printf("camera:(%f,%f,%f)\n", d_cameraPos.x, d_cameraPos.y, d_cameraPos.z);
	printf("eoc:(%f,%f,%f)\n", eoc_pos.x, eoc_pos.y, eoc_pos.z);*/

	//for (int i = 0; i < 4; i++)
	//	printf("(%f,%f,%f,%f)\n", d_modelView[4 * i + 0], d_modelView[4 * i + 1], d_modelView[4 * i + 2], d_modelView[4 * i + 3]);
	for (int x = beginX; x < top; x++)
	{
		float ratio = (x * 1.0f - beginX*1.0f) / (top - 1 - beginX);
		float3 realPos = projective_interpo(beforePos, endPos, ratio);
		int index = y*d_outTextureWidth + x;
		float dis = distance(leftEdgePos, realPos, eoc_pos);
		d_cudaTexture[index] = make_float4(dis, realPos.x, realPos.y, realPos.z);
	}
}

__device__ void FillSpan(int beginX, int endX, int y, float2 beginUv, float2 endUv)
{
	int top = min(endX, d_outTextureWidth);
	for (int x = beginX; x < top; x++)
	{
		int index = y*d_outTextureWidth + x;
		float uvx = beginUv.x + (endUv.x - beginUv.x)*(x - beginX) / (top - beginX);
		d_cudaTexture[index] = tex2D(cudaColorTex, uvx, beginUv.y);
	}
}
__global__ void renderToTexutre(int kernelWidth, int kernelHeight)
{
	int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (y > kernelHeight)
		return;
	//if (y != 807)
	//	return;
	int listIndex = y;
	int rowLength = imageWidth;
	ListNote currentNote = *((ListNote*)&d_cudaPboBuffer[listIndex]);
	int texEnd = 0;
	int texBegin = 0;
	int fillBegin = 0;
	int fillEnd = 0;
	int acuumPixel = 0, span = 0;
	//("begin:%d,end%d,index:%d\n", currentNote.beginIndex, currentNote.endIndex, currentNote.nextPt);*	printf("init:%d\n", d_cudaPboBuffer[listIndex].x);
	/*while (currentNote.nextPt != 0)
	{
	currentNote = d_listBuffer[currentNote.nextPt];
	rowLength += currentNote.endIndex - currentNote.beginIndex;
	//printf("begin:%d,end%d,index:%d,length:%d\n", currentNote.beginIndex, currentNote.endIndex, currentNote.nextPt,rowLength);
	}*/
	//printf("printf:%d\n", rowLength);
	float factor = imageWidth*1.0 / rowLength;
	currentNote = *((ListNote*)&d_cudaPboBuffer[listIndex]);
	int leftEdgeIndex = 0;
	while (currentNote.nextPt != 0)
	{


		currentNote = d_listBuffer[currentNote.nextPt];
		//printf("current:b:%d,e:%d,n:%d,leftEdge:%d\n", currentNote.beginIndex, currentNote.endIndex, currentNote.nextPt, currentNote.leftEdge);

		texEnd = currentNote.endIndex;
		span = currentNote.endIndex - currentNote.beginIndex;
		leftEdgeIndex = currentNote.leftEdge;
		fillBegin = texBegin + acuumPixel;
		fillEnd = texEnd + acuumPixel;
		FillSpan(fillBegin*factor, fillEnd*factor, y, toUv(texBegin, y), toUv(texEnd, y));  //for 循环，左闭右开
		FillVolumn((fillEnd)*factor, (fillEnd + span)*factor, y, texEnd, leftEdgeIndex);

		acuumPixel += span;
		texBegin = currentNote.endIndex;
		//printf("texBegin:%d,acuumPixel:%d,n:%d\n", texBegin, acuumPixel);

	}
	fillBegin = texBegin + acuumPixel;
	//printf("final:(%d,%d) u(%f,%f)\n", fillBegin, imageWidth + span, toUv(texBegin, y).x, toUv(imageWidth - 1, y).x);
	FillSpan(fillBegin*factor, (imageWidth + acuumPixel)*factor, y, toUv(texBegin, y), toUv(imageWidth - 1, y));


}
ListNote *device_data, *device_top_data = NULL;
int atomBuffer = 1;
#ifdef DEBUG
ListNote *host_data = NULL;
#endif
extern void cudaInit(int height, int width, int k, int rowLarger)
{
	checkCudaErrors(cudaMalloc(&device_data, height*k*sizeof(ListNote)));

	checkCudaErrors(cudaMemcpyToSymbol(d_listBuffer, &device_data, sizeof(ListNote*)));

	checkCudaErrors(cudaMemcpyToSymbol(d_atomic, &atomBuffer, sizeof(int)));
	checkCudaErrors(cudaMemset(device_data, 0, height*k*sizeof(ListNote)));
	//checkCudaErrors(cudaMemset(cuda_TexturePbo_buffer, 0, width* height*rowLarger*sizeof(float4)));
#ifdef DEBUG
	checkCudaErrors(cudaMallocHost(&host_data, height*k*sizeof(ListNote)));
#endif
	checkCudaErrors(cudaMalloc(&modelView, 16 * sizeof(float)));

	//host_data = (ListNote*)malloc(height*k*sizeof(ListNote));
	//memset(host_data, 0, height*k*sizeof(ListNote));
	//checkCudaErrors(cudaMemcpy((void *)device_data, (void *)host_data, height * k * sizeof(ListNote), cudaMemcpyDeviceToHost));


}
extern "C" void countRow(int width, int height, Camera * pCamera, Camera * pEocCam, Camera * pEocTopCamera)
{
	checkCudaErrors(cudaMemset(cuda_TexturePbo_buffer, 0, ROWLARGER*width*height*sizeof(float4)));
	checkCudaErrors(cudaMemset(cuda_top_TexturePbo_buffer, 0, ROWLARGER*width*ROWLARGER*height*sizeof(float4)));

	checkCudaErrors(cudaMemcpyToSymbol(d_cameraPos, &pCamera->getCameraPos(), 3 * sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(d_eocPos, &pEocCam->getCameraPos(), 3 * sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(d_eocTopPos, &pEocTopCamera->getCameraPos(), 3 * sizeof(float)));

	checkCudaErrors(cudaMemcpy(modelView, pCamera->getModelViewMat(), 16 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_modelView, &modelView, sizeof(float*)));
	cudaEvent_t begin_t, end_t;
	checkCudaErrors(cudaEventCreate(&begin_t));
	checkCudaErrors(cudaEventCreate(&end_t));

	cudaEventRecord(begin_t, 0);

	checkCudaErrors(cudaMemcpyToSymbol(d_atomic, &atomBuffer, sizeof(int)));
	checkCudaErrors(cudaMemset(cuda_PBO_Buffer, 0, height*sizeof(ListNote)));

	dim3 blockSize(1, 16, 1);
	dim3 gridSize(1, height / blockSize.y, 1);
	countRowKernel << <gridSize, blockSize >> >(1, height);
	cudaEventRecord(end_t, 0);
	cudaEventSynchronize(end_t);
	float costtime;
	checkCudaErrors(cudaEventElapsedTime(&costtime, begin_t, end_t));

	renderToTexutre << <gridSize, blockSize >> >(1, height);
	/**/
	//top Camera
	checkCudaErrors(cudaMemcpyToSymbol(d_atomic, &atomBuffer, sizeof(int)));
	checkCudaErrors(cudaMemset(cuda_PBO_Buffer, 0, height*sizeof(ListNote)));

	dim3 gridSizeT(1, ROWLARGER* width / blockSize.y, 1);
	countRowKernelTop << <gridSizeT, blockSize >> >(width, 1);
	renderToTexutreTop << <gridSizeT, blockSize >> >(ROWLARGER*width, 1);

	checkCudaErrors(cudaEventDestroy(begin_t));
	checkCudaErrors(cudaEventDestroy(end_t));

#ifdef DEBUG
	/*
	int arraySize = 0;
	checkCudaErrors(cudaMemcpy((void *)host_data, (void *)device_data, height*10*sizeof(ListNote), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpyFromSymbol(&arraySize, d_atomic, sizeof(int)));

	for (int i = 0; i < arraySize; i++)
	{
	printf("b:%d,n:%d,next:%d\n", host_data[i].beginIndex, host_data[i].endIndex, host_data[i].nextPt);
	}*/
#endif

}

extern "C"  void cudaRelateTex(CudaTexResourse * pResouce)
{

	cudaArray *tmpcudaArray;
	cudaGraphicsResource ** pCudaTex = pResouce->getResPoint();
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&tmpcudaArray, *pCudaTex, 0, 0));
	int w = pResouce->getWidth();
	int h = pResouce->getHeight();
	checkCudaErrors(cudaMemcpyToSymbol(imageWidth, &w, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(imageHeight, &h, sizeof(int)));
	if (occluderbuffer_t == pResouce->getType())
	{
		checkCudaErrors(cudaBindTextureToArray(cudaOccuderTex, tmpcudaArray, channelDesc));
		cudaOccuderTex.filterMode = cudaFilterModeLinear;
	}
	else if (edgebuffer_t == pResouce->getType())
	{
		checkCudaErrors(cudaBindTextureToArray(cudaProgTex, tmpcudaArray, channelDesc));
		cudaProgTex.filterMode = cudaFilterModePoint;
	}
	else if (color_t == pResouce->getType())
	{
		checkCudaErrors(cudaBindTextureToArray(cudaColorTex, tmpcudaArray, channelDesc));
		cudaColorTex.filterMode = cudaFilterModePoint;
	}
	else if (pos_t == pResouce->getType())
	{
		checkCudaErrors(cudaBindTextureToArray(cudaPosTex, tmpcudaArray, channelDesc));
		cudaPosTex.filterMode = cudaFilterModePoint;

	}
	else if (occluderTopbuffer_t == pResouce->getType())
	{
		checkCudaErrors(cudaBindTextureToArray(cudaTopOccuderTex, tmpcudaArray, channelDesc));
		cudaTopOccuderTex.filterMode = cudaFilterModePoint;
	}

}
extern "C" void cudaRelateArray(CudaPboResource * pResource)
{
	size_t numBytes;
	cudaGraphicsResource ** pCudaTex = pResource->getResPoint();
	int w = pResource->getWidth();
	int h = pResource->getHeight();
	if (list_e == pResource->getType())
	{
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&cuda_PBO_Buffer, &numBytes, *pCudaTex));
		checkCudaErrors(cudaMemcpyToSymbol(d_cudaPboBuffer, &cuda_PBO_Buffer, sizeof(ListNote*)));
	}
	else if (float4_t == pResource->getType())
	{
		checkCudaErrors(cudaMemcpyToSymbol(d_outTextureWidth, &w, sizeof(int)));
		checkCudaErrors(cudaMemcpyToSymbol(d_outTextureHeigh, &h, sizeof(int)));

		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&cuda_TexturePbo_buffer, &numBytes, *pCudaTex));
		checkCudaErrors(cudaMemcpyToSymbol(d_cudaTexture, &cuda_TexturePbo_buffer, sizeof(float4*)));
	}
	else  if (top_float4_t == pResource->getType())
	{
		checkCudaErrors(cudaMemcpyToSymbol(d_outTopTextureWidth, &w, sizeof(int)));
		checkCudaErrors(cudaMemcpyToSymbol(d_outTopTextureHeight, &h, sizeof(int)));

		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&cuda_top_TexturePbo_buffer, &numBytes, *pCudaTex));
		checkCudaErrors(cudaMemcpyToSymbol(d_cudaTopTexture, &cuda_top_TexturePbo_buffer, sizeof(float4*)));

	}
}

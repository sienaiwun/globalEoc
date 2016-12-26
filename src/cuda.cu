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
texture<float4, 2, cudaReadModeElementType> optixColorTex;
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
__device__ int d_imageWidth, d_imageHeight, d_outTextureWidth, d_outTextureHeigh, d_outTopTextureWidth, d_outTopTextureHeight,d_construct_width,d_construct_height;
__device__ int d_index;
__device__ ListNote* d_listBuffer;
__device__ ListNote* d_listBuffer_top;
__device__ int d_atomic;
__device__ float3 d_cameraPos;
__device__ float3 d_eocPos;
__device__ float3 d_eocTopPos;
float* modelView;
__device__ float* d_modelView;
float* proj;
__device__ float* d_porj;


float* modelView_construct;
float* project_construct;
float* modelView_inv;
__device__ float* d_modelView_construct;
__device__ float* d_project_construct;
__device__ float3 d_construct_cam_pos;
__device__ float* d_modeView_inv_construct;
__device__ float2 d_bbmin, d_bbmax;


__device__ float4* d_cuda_construct_texture;
float4 *cuda_construct_texturePbo_buffer;

__device__ float4* d_map_buffer;
float4* cuda_map_buffer;
__host__ __device__ float4 MutiMatrix(float * Matrix, float4 invalue)
{
	float x = invalue.x;
	float y = invalue.y;
	float z = invalue.z;
	float w = invalue.w;

	float outx = x*Matrix[0] + y*Matrix[4] + z*Matrix[8] + w*Matrix[12];
	float outy = x*Matrix[1] + y*Matrix[5] + z*Matrix[9] + w*Matrix[13];
	float outz = x*Matrix[2] + y*Matrix[6] + z*Matrix[10] + w*Matrix[14];
	float outw = x*Matrix[3] + y*Matrix[7] + z*Matrix[11] + w*Matrix[15];

	return make_float4(outx, outy, outz, outw);
}
__host__ __device__ float  element(float* _array,int row, int col) 
{
	return _array[row | (col << 2)];
}
__host__ __device__ float4 MutiMatrixN(float * Matrix, float4 invalue)
{
	float x = invalue.x;
	float y = invalue.y;
	float z = invalue.z;
	float w = invalue.w;
	float r[4];
	for (int i = 0; i < 4; i++)
	{
		r[i] = (x * element(Matrix, i, 0) + y * element(Matrix, i, 1) + z * element(Matrix, i, 2) + w * element(Matrix, i, 3));
	}
	return make_float4(r[0], r[1], r[2], r[3]);
}
__host__ __device__ void MutiMatrix(float* src, float* matrix, float* r)
{
		for (int i = 0; i < 4; i++)
		{			
			r[i] = (src[0] * element(matrix, i, 0) + src[1] * element(matrix,i, 1) + src[2] * element(matrix, i, 2) + src[3] * element(matrix, i, 3));
		}

}
__device__ int2 nearestTc(float2 tc)
{
	return make_int2(tc.x, tc.y);//直接进行int转换，因为减去0.5+0.5
}
__device__ float4 colorTextreNorTc(float2 tc)
{
	float2 nonNorTc = tc* make_float2(d_imageWidth,d_imageHeight);
	int2 mapTx = nearestTc(nonNorTc);
	int index = mapTx.y * d_imageWidth + mapTx.x;
	int mappedX = (int)(d_map_buffer[index].x+0.5);

	//printf("searched tc:(%d,%f),z:(%f)\n", mappedX, nonNorTc.y, tex2D(cudaColorTex, nonNorTc.x, nonNorTc.y).z);
	return tex2D(optixColorTex, mappedX, nonNorTc.y);
}
__device__ bool canGetMappedPosition(float2 tc,float4* poc)
{
	float2 nonNorTc = tc* make_float2(d_imageWidth, d_imageHeight);
	int2 mapTx = nearestTc(nonNorTc);
	int index = mapTx.y * d_imageWidth + mapTx.x;
	int mappedY = (int)(d_map_buffer[index].y+ 0.5);
	if (mappedY < 1)
	{
		return false;
	}
	//printf("mapped coord:(%d,%d)\n", mappedY, mapTx.y);
	*poc = tex2D(optixColorTex, mappedY, nonNorTc.y);
	return true;

}
__host__ __device__ void MutiMatrix(float * Matrix, float x, float y, float z, float &outx, float &outy, float &outz)
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
	for (int y = 0; y< d_imageHeight; y++)
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
	for (int x = 0; x < d_imageWidth; x++)
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
	if (index > d_imageWidth)
	{
		FillLine(index);
		return;
	}
//	if (index != 512)
//		return;
	int listIndex = index;
	int rowLength = d_imageWidth;
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
	float factor = d_imageWidth*1.0 / rowLength;
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
	//printf("final:(%d,%d) u(%f,%f)\n", fillBegin, d_imageWidth + span, toUv(index, texBegin).y, toUv(index,d_imageWidth).y);
	FillSpanTop(fillBegin*factor, (d_imageWidth + acuumPixel)*factor, index, toUv(index,texBegin ), toUv(index, d_imageWidth - 1));

}
// 在第y 行的beginX 到endX直接记录中空的区域的位置信息leftEdge是用来评估左边边界，endUv是用来算左右两边的深度插值
__device__ void FillVolumn(int beginX, int endX, int y, int endUv, int leftEdge,int accumIndex )
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
		int lenght = (top - 1 - beginX);
		float ratio = (x * 1.0f - beginX*1.0f) / (top - 1 - beginX);
		float3 realPos = projective_interpo(beforePos, endPos, ratio);
		int index = y*d_outTextureWidth + x;
		float dis = distance(leftEdgePos, realPos, eoc_pos);
		d_cudaTexture[index] = make_float4(dis, realPos.x, realPos.y, realPos.z);
		int originMappos = y*d_imageWidth + accumIndex - lenght;
		d_map_buffer[originMappos].y = x;
		accumIndex++;
	}
}
// 在第y 行的beginX 到endX 的这一块区域插入原来图像从beginUV到endUV的这一横条图像accumIndexX 记录原来图像到新图像的映射
__device__ void FillSpan(int beginX, int endX, int y, float2 beginUv, float2 endUv,int* accumIndexX)
{
	int top = min(endX, d_outTextureWidth);
	for (int x = beginX; x < top; x++)
	{
		int index = y*d_outTextureWidth + x;
		float uvx = beginUv.x + (endUv.x - beginUv.x)*(x - beginX) / (top - beginX);
		d_cudaTexture[index] = tex2D(cudaColorTex, uvx, beginUv.y);
		int originMappos = y*d_imageWidth + *accumIndexX;
		d_map_buffer[originMappos].x = x;
		*accumIndexX += 1;
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
	int rowLength = d_imageWidth;
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
	int accum_index = 0;  // 记录累计
	float factor = d_imageWidth*1.0 / rowLength;
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
		FillSpan(fillBegin*factor, fillEnd*factor, y, toUv(texBegin, y), toUv(texEnd, y), &accum_index);  //for 循环，左闭右开
		FillVolumn((fillEnd)*factor, (fillEnd + span)*factor, y, texEnd, leftEdgeIndex, accum_index);

		acuumPixel += span;
		texBegin = currentNote.endIndex;
		//printf("texBegin:%d,acuumPixel:%d,n:%d\n", texBegin, acuumPixel);

	}
	fillBegin = texBegin + acuumPixel;
	//printf("final:(%d,%d) u(%f,%f)\n", fillBegin, d_imageWidth + span, toUv(texBegin, y).x, toUv(d_imageWidth - 1, y).x);
	FillSpan(fillBegin*factor, (d_imageWidth + acuumPixel)*factor, y, toUv(texBegin, y), toUv(d_imageWidth - 1, y), &accum_index);


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
	checkCudaErrors(cudaMalloc(&proj, 16 * sizeof(float)));
	
	checkCudaErrors(cudaMalloc(&cuda_map_buffer, width*height * sizeof(float4)));
	checkCudaErrors(cudaMemcpyToSymbol(d_map_buffer, &cuda_map_buffer, sizeof(float*)));

	//host_data = (ListNote*)malloc(height*k*sizeof(ListNote));
	//memset(host_data, 0, height*k*sizeof(ListNote));
	//checkCudaErrors(cudaMemcpy((void *)device_data, (void *)host_data, height * k * sizeof(ListNote), cudaMemcpyDeviceToHost));

	
	

}
extern "C" void countRow(int width, int height, Camera * pCamera, Camera * pEocCam, Camera * pEocTopCamera)
{
	checkCudaErrors(cudaMemset(cuda_TexturePbo_buffer, 0, ROWLARGER*width*height*sizeof(float4)));
	checkCudaErrors(cudaMemset(cuda_top_TexturePbo_buffer, 0, ROWLARGER*width*ROWLARGER*height*sizeof(float4)));
	checkCudaErrors(cudaMemset(cuda_map_buffer, 0, width*height*sizeof(float4)));

	checkCudaErrors(cudaMemcpyToSymbol(d_cameraPos, &pCamera->getCameraPos(), 3 * sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(d_eocPos, &pEocCam->getCameraPos(), 3 * sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(d_eocTopPos, &pEocTopCamera->getCameraPos(), 3 * sizeof(float)));

	checkCudaErrors(cudaMemcpy(modelView, pCamera->getModelViewMat(), 16 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_modelView, &modelView, sizeof(float*)));
	checkCudaErrors(cudaMemcpy(proj, pCamera->getProjection(), 16 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_porj, &proj, sizeof(float*)));


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
	
		checkCudaErrors(cudaMemcpyToSymbol(d_imageWidth, &w, sizeof(int)));
		checkCudaErrors(cudaMemcpyToSymbol(d_imageHeight, &h, sizeof(int)));
		checkCudaErrors(cudaBindTextureToArray(cudaColorTex, tmpcudaArray, channelDesc));
		cudaColorTex.filterMode = cudaFilterModePoint;
	}
	else if (pos_t == pResouce->getType())
	{
		checkCudaErrors(cudaMemcpyToSymbol(d_imageWidth, &w, sizeof(int)));
		checkCudaErrors(cudaMemcpyToSymbol(d_imageHeight, &h, sizeof(int)));
		checkCudaErrors(cudaBindTextureToArray(cudaPosTex, tmpcudaArray, channelDesc));
		cudaPosTex.filterMode = cudaFilterModePoint;

	}
	else if (occluderTopbuffer_t == pResouce->getType())
	{
		checkCudaErrors(cudaBindTextureToArray(cudaTopOccuderTex, tmpcudaArray, channelDesc));
		cudaTopOccuderTex.filterMode = cudaFilterModePoint;
	}
	else if (optixColorTex_t == pResouce->getType())
	{
		checkCudaErrors(cudaBindTextureToArray(optixColorTex, tmpcudaArray, channelDesc));
		optixColorTex.filterMode = cudaFilterModePoint;
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
	else if (construct_t == pResource->getType())
	{
		checkCudaErrors(cudaMemcpyToSymbol(d_construct_width, &w, sizeof(int)));
		checkCudaErrors(cudaMemcpyToSymbol(d_construct_height, &h, sizeof(int)));

		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&cuda_construct_texturePbo_buffer, &numBytes, *pCudaTex));
		checkCudaErrors(cudaMemcpyToSymbol(d_cuda_construct_texture, &cuda_construct_texturePbo_buffer, sizeof(float4*)));
	}
}
void mapConstruct(Camera * pReconstructCamer)
{

	checkCudaErrors(cudaMemcpyToSymbol(d_construct_cam_pos, &pReconstructCamer->getCameraPos(), 3 * sizeof(float)));
	checkCudaErrors(cudaMemcpy(modelView_construct, pReconstructCamer->getModelViewMat(), 16 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_modelView_construct, &modelView_construct, sizeof(float*)));
	checkCudaErrors(cudaMemcpy(project_construct, pReconstructCamer->getProjection(), 16 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_project_construct, &project_construct, sizeof(float*)));
	nv::matrix4f invModelView = inverse(nv::matrix4f(pReconstructCamer->getModelViewMat()));
	checkCudaErrors(cudaMemcpy(modelView_inv, invModelView.get_value(), 16 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_modeView_inv_construct, &modelView_inv, sizeof(float*)));

	nv::vec2f bbmin = nv::vec2f(pReconstructCamer->getImageMin().x, pReconstructCamer->getImageMin().y);
	nv::vec2f bbmax = nv::vec2f(pReconstructCamer->getImageMax().x, pReconstructCamer->getImageMax().y);
	checkCudaErrors(cudaMemcpyToSymbol(d_bbmin, &bbmin, 2*sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(d_bbmax, &bbmax, 2*sizeof(float)));
	nv::vec2f tc = nv::vec2f(412, 512) / nv::vec2f(1024.0, 1024.0);
	nv::vec2f xy = bbmin + (bbmax - bbmin)*tc;
	nv::vec4f temp;
	MutiMatrix((float*)nv::vec4f(xy,-1,1), (float*)invModelView.get_value(), (float*)& temp);// *;
	//MutiMatrix((float*)&modelViewValue, (float*)invModelView.get_value(), (float*)& temp);// *;

	// nv::vec4f fuck = invModelView * nv::vec4f(modelViewValue);
	//nv::vec4f beginPoint = nv::vec4f(15.402321, -17.913536, -52.650398,1);

	//beginPoint = nv::vec4f(-4.31230021, -17.7475834, -47.2878876, 1);
	//beginPoint = nv::vec4f( nv::vec3f(temp), 1);
	nv::vec4f temp1 = nv::matrix4f(pReconstructCamer->getModelViewMat())* temp;
	nv::vec4f final = nv::matrix4f(pReconstructCamer->getProjection())* temp1;
	final /= final.w;
	final.x = final.x*0.5 + 0.5;
	final.y = final.y*0.5 + 0.5;
	final.x *= 1024;
	final.y *= 1024;

	
}

__device__ float3 getImagePos(float2 tc)
{
	float2 xy = d_bbmin + (d_bbmax - d_bbmin)*tc;
	xy = xy;
	float4 temp = MutiMatrixN(d_modeView_inv_construct, make_float4(xy.x, xy.y, -1, 1));// must
	temp = temp / temp.w;
	return make_float3(temp.x, temp.y, temp.z);
}
__device__ float3 toFloat3(float4 inValue)
{
	return make_float3(inValue.x / inValue.w, inValue.y / inValue.w, inValue.z / inValue.w);
}
__device__ bool isTracingEdge(float2 tc)
{
	float2 nonNorTc = tc* make_float2(d_imageWidth, d_imageHeight);
	return abs(tex2D(cudaProgTex, nonNorTc.x, nonNorTc.y).x) > 0.05 || abs(tex2D(cudaProgTex, nonNorTc.x, nonNorTc.y).y) > 0.05;
}

 __device__ int intersectTexRay(float3 posW, float3 directionW,	float4& oc)
{

	float2 d_mapScale = 1.0/make_float2(d_construct_width, d_construct_height);
	float3 rayStart, rayEnd;
	float4 color;
	//printf("posW:(%f,%f,%f,1)\n", posW.x, posW.y, posW.z);
	float4 posWE = MutiMatrixN(d_modelView,make_float4(posW, 1));
	float4 temp2 = MutiMatrixN(d_porj, posWE);
	temp2 = temp2 / temp2.w;
	//printf("temp2:%f,%f", (temp2.x*0.5 + 0.5) * 1024, (temp2.y*0.5 + 0.5) * 1024);
	float3 posW3 = toFloat3(posWE);
	float4 temp = MutiMatrixN(d_modelView_construct, make_float4(directionW, 0));
	float3 RE = normalize(make_float3(temp.x, temp.y, temp.z));
	//printf("RE:(%f,%f,%f,1)\n", RE.x, RE.y, RE.z);
	float epison = 10.2;
	rayStart = posW3 + RE*epison;

	float max_rfl = 370;//far*diffuseColor.w;
	rayEnd = posW3 + RE*max_rfl;

	//p.color0.xy = tc;
	if (rayEnd.z>0)
	{
		float step = -posW3.z / RE.z;
		rayEnd = posW3 + RE*(step - 1);
	}

	temp;
	temp = MutiMatrixN(d_porj, make_float4(rayStart, 1));

	float3 projStart = toFloat3(temp);
	temp = MutiMatrixN(d_porj, make_float4(rayEnd, 1));
	float3 projEnd = toFloat3(temp);

	projStart.x = 0.5*projStart.x + 0.5;
	projEnd.x = 0.5*projEnd.x + 0.5;
	projStart.y = 0.5*projStart.y + 0.5;
	projEnd.y = 0.5*projEnd.y + 0.5;

	//printf("projStart(%f,%f),projEnd(%f,%f)\n", (projStart.x) * 1024, (projStart.y) * 1024, projEnd.x * 1024, projEnd.y * 1024);
	rayStart.z = rayStart.z;

	if (projStart.x>1 || projStart.x<0 || projStart.y<0 || projStart.y>1 || rayStart.z>0)
	{
		return 0;
	}
	//oc = make_float4(projStart.x,projStart.y,projStart.z,0.7);	
	//return 1;
	float alpha = 0;
	float2 interval = make_float2(projEnd.x, projEnd.y) - make_float2(projStart.x, projStart.y);
	int stepN;
	//printf("interval:(%f,%f)\n", interval.x, interval.y);
	//printf("1024*d_mapScale:(%f,%f)\n", d_mapScale.x*1024,d_mapScale.y*1024);
	if (abs(interval.x)>abs(interval.y))
		stepN = abs(interval.x) / d_mapScale.x + 1;
	else
		stepN = abs(interval.y) / d_mapScale.y + 1;
	//printf("stepN:%d\n", stepN);
	float currSamplePointZ, currRayPointZ, prevSamplePointZ, prevRayPointZ;
	float3 currSamplePoint, currRayPoint;
	int n = 0;
	float2 tc = make_float2(projStart.x, projStart.y);
	bool isNotValid = true;

	if (tc.x>1 || tc.x<0 || tc.y<0 || tc.y>1)
	{
		return 0;
	}
	n = 0;
	int count = 0;
	bool formerCompare = false;
	bool compare = false;
	float2 formertc;
	//printf("interval*1024(%f,%f)\n", (interval.x) * 1024, (interval.y) * 1024);
	bool isBelowO = true;
	//printf("stepN:%d\n", stepN);
	if (stepN<2)
	{
		//printf("here");
		oc = colorTextreNorTc(tc + interval / 2);
		return 1;
	}
	while (tc.x >= 0 && tc.x <= 1 && tc.y >= 0 && tc.y <= 1 && n <= stepN)
	{
		alpha = (float) n / stepN;
		currRayPointZ = 1 / ((1 - alpha)*(1 / rayStart.z) + (alpha)*(1 / rayEnd.z));
		currSamplePointZ = colorTextreNorTc(tc).w;
		//printf("tc:(%f,%f),n:%d,stepN:%d,z:(%f,%f)\n", 1024 * tc.x, 1024 * tc.y, n, stepN, currRayPointZ, currSamplePointZ);
		if ((currSamplePointZ<0) && (currRayPointZ <= currSamplePointZ))
		{
			
				
				color = colorTextreNorTc(tc);
				float lastAlpha =0;
				if (n >= 1)
					lastAlpha = (float)(n - 1) / stepN;
				float2 lastTc = make_float2(projStart.x, projStart.y) + interval* lastAlpha;
				if (isTracingEdge(lastTc))
				{
					int localN = n;
					float4 localcolor;
					bool getColor = canGetMappedPosition(tc, &localcolor);
					while (getColor)
					{
						float localalpha = (float)localN / stepN;
						float2 localTc = make_float2(projStart.x, projStart.y) + interval* localN / stepN;
						getColor = canGetMappedPosition(localTc, &localcolor);
						float localRayPointZ = 1 / ((1 - localalpha)*(1 / rayStart.z) + (localalpha)*(1 / rayEnd.z));
						float localSamplePointZ = localcolor.w;
						//printf("tc:(%f,%f),z:(%f,%f)\n", 1024 * localTc.x, 1024 * localTc.y, localRayPointZ, localSamplePointZ);
						if ((localRayPointZ < 0) && (localRayPointZ <= localSamplePointZ))
						{
							//printf("found tc:(%f,%f),z:(%f,%f)\n", 1024 * localTc.x, 1024 * localTc.y, localRayPointZ, localSamplePointZ);
							oc = localcolor;
							return 1;
						}
					    localN ++;
					   
					};
					/*
					  进行隐藏区域的光线跟踪
					*/
					
				}
				color.w = 1;
				oc = color;
				//printf("found");
				return 1;
			
		}
		n += 1;
		tc = make_float2(projStart.x, projStart.y) + interval* n / stepN;
	}
	return 0;

}

__global__ void construct_kernel(int kernelWidth, int kernelHeight)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if (x >= kernelWidth || y >= kernelHeight)
		return;
 //	if (x != 487 || y != 514)
//		return;
	const int index = y*kernelWidth + x;
	float2 tc = make_float2(x + 0.5, y + 0.5) / make_float2(kernelWidth, kernelHeight);
	float3 beginPoint = getImagePos(tc) ;
	//printf("beginPoint:(%f,%f,%f)\n", beginPoint.x, beginPoint.y, beginPoint.z);
	float3 viewDirection = getImagePos(tc) - d_construct_cam_pos;

	//printf("viewDirection:(%f,%f,%f)\n", viewDirection.x, viewDirection.y, viewDirection.z);
	float4 outColor;
	if (intersectTexRay(beginPoint,viewDirection,outColor))
	{
	//	printf("here outcolor(%f,%f,%f,%f)\n", outColor.x, outColor.y, outColor.z, outColor.w);
		d_cuda_construct_texture[index] = make_float4(outColor.x, outColor.y, outColor.z,1);//tex2D(cudaColorTex, x, y);
	}
	else
	{ 
		d_cuda_construct_texture[index] = make_float4(0,0,0, 1);//tex2D(cudaColorTex, x, y);
	}
}
void construct_cudaInit()
{
	checkCudaErrors(cudaMalloc(&modelView_construct, 16 * sizeof(float)));
	checkCudaErrors(cudaMalloc(&project_construct, 16 * sizeof(float)));
	checkCudaErrors(cudaMalloc(&modelView_inv, 16 * sizeof(float)));
		

}
void cuda_Construct(int width, int height)
{
	dim3 blockSize(16, 16, 1);
	dim3 gridSize(width / blockSize.x, height / blockSize.y, 1);
	construct_kernel << <gridSize, blockSize >> >(width, height);
}

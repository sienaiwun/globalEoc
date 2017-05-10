
#include "cuda.h"
#include <helper_math.h>
#include <nvMatrix.h>
#include "Camera.h"
#define BEGINOFFSET (1.4f)
#define ENDOFFSET (370)
//#define PRINTDEBUG
#ifdef PRINTDEBUG
#define my_printf(...) \
	printf(__VA_ARGS__)
#else
#define my_printf(...) ;
#endif

#define BEGINDEPTH 0
#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif



#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define INTERSECT 1
#define OHTEROBJECT (-1)
#define MISSINGNOTE 0
#define NOOBJECTNOTE 2
#define OUTOCCLUDED 3
#define FLT_MAX 99999.9
		 //求交的宏
#define RAYISUP 0
#define RAYOUT 1
#define RAYISUNDER 2
#define RAYBACKTOMAIN 3
#define RAYVALID 4
#define RAYBEGIN 5
		 texture<float4, 2, cudaReadModeElementType> cudaEdgeTex;//记录的是Edge,edge里面x记录x的sobal,edge里面y记录y的sobal
texture<float4, 2, cudaReadModeElementType> cudaOccuderTex;
texture<float4, 2, cudaReadModeElementType> cudaTopOccuderTex;
texture<float4, 2, cudaReadModeElementType> cudaColorTex;
texture<float4, 2, cudaReadModeElementType> cudaPosTex;
texture<float4, 2, cudaReadModeElementType> cudaNormalTex;
texture<float4, 2, cudaReadModeElementType> optixColorTex;    //记录的是optix 追踪的eoc图像，w通道记录的是投影深度，原视点相机为负值，负值为正值，我也不知道为啥
texture<float4, 2, cudaReadModeElementType> posBlendTex;   //记录的是eoc相机空间下的位置图像，w通道记录的是是否是边界
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
typedef enum {
	isVolumn,
	notVolumn,
}pixelEnum;



uint3 *cuda_PBO_Buffer;
uint3 *cuda_PBO_Top_Buffer;

__device__ uint3* d_cudaPboBuffer;
__device__ uint3* d_cudaPboTopBuffer;
float4 *cuda_TexturePbo_buffer, *cuda_top_TexturePbo_buffer;
__device__ float4* d_cudaTexture;
__device__ float4* d_cudaTopTexture;
__device__ int d_imageWidth, d_imageHeight, d_outTextureWidth, d_outTextureHeigh, d_outTopTextureWidth, d_outTopTextureHeight, d_construct_width, d_construct_height;
__device__ int d_index;
__device__ ListNote* d_listBuffer;
__device__ ListNote* d_listBuffer_top;
__device__ int d_atomic;
__device__ int d_atomic_top;
__device__ float3 d_cameraPos;
__device__ float3 d_eocPos;
__device__ float3 d_eocTopPos;
__device__ float3 d_rightRD;
__device__ float3 d_topRD;
float* modelView, *modelView_inv,*modelView_construct_inv;
__device__ float* d_modelView,*d_modelView_inv;
float* proj,*proj_inv;
__device__ float* d_proj, *d_proj_inv;


float* modelViewRight;
__device__ float* d_modelViewRight;

float* modelViewTop;
__device__ float* d_modelViewTop;

float* modelView_construct;
float* project_construct;

__device__ float* d_modelView_construct;
__device__ float* d_project_construct;
__device__ float3 d_construct_cam_pos;
__device__ float* d_modeView_inv_construct;
__device__ float2 d_bbmin, d_bbmax;


__device__ float4* d_cuda_construct_texture;
float4 *cuda_construct_texturePbo_buffer;
struct occludedInfo
{
public:
	int m_beginLineNum,m_endLineNum;
	int m_stepNum;
	bool m_isOccluded;
	occludedInfo(bool is) :m_isOccluded(is)
	{
		m_stepNum = 0;
	}
};
struct EOCPixel
{
	int m_state;
	float m_detpth;
	bool m_isValid;
	bool m_isAtEdge;
	__device__ EOCPixel() :m_isAtEdge(true), m_isValid(false)
	{
	};
	__device__ EOCPixel(int state, float depth) :m_state(state), m_detpth(depth)
	{
	};
};
__device__ float4* d_map_buffer;  //d_map_buffer x 记录的是texture 到新的texture的映射，y记录的是遮挡像素的地区到新扩增的地区的映射，z记录的是遮挡地区的noteId
float4* cuda_map_buffer;
#define EOC_RIGHT_EDGE_VALID 1
#define EOC_RIGHT_TOP_VALID 2
struct edgeInfo
{
	bool isRightEdge;
	float dzdx;
	__device__ __host__ edgeInfo():	isRightEdge (false),dzdx(0)
	{
	
	}
};
__device__ edgeInfo* d_edge_buffer;// 
edgeInfo* edgeBuffer;
__device__ int nearestInt(float value)
{
	return value + 0.5;
}
__device__ float repo(float value)
{
	return 1.0f / value;
}
__device__ int startInt(float value, bool isUp)
{
	if (isUp)
		return floor(value);
	else
		return ceil(value);
}
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
__host__ __device__ float  element(float* _array, int row, int col)
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
		r[i] = (src[0] * element(matrix, i, 0) + src[1] * element(matrix, i, 1) + src[2] * element(matrix, i, 2) + src[3] * element(matrix, i, 3));
	}

}
__device__ float getRatioInSpan(float3 beginPos, float3 endPos, float* p_modelView, float3 testPos);
__device__ float3 projective_interpo(float3 beginPos, float3 endPos, float* p_modelView, float ratio, int print);
/*
记录光线和三角形求交的结果，求交的结果的z值比例记录在 proj_ratio 上
*/
__device__ bool rayIntersertectTriangle(float3 origin, float3 directionN, float3 cameraPos, float3 edgePoint1/*beginPos*/, float3 edgePoint2/*endPos*/, float* modelView_float, float noteSpan, float3* pIntersectWorld3, float3* pLineIntersect, bool& isOnrTiangle, float& proj_ratio, float3& reversePoint3)
{
	//printf("origin:(%f,%f,%f)\n", origin.x, origin.y, origin.z);
	//printf("directionN:(%f,%f,%f)\n", directionN.x, directionN.y, directionN.z);

	const float3 e0 = edgePoint1 - cameraPos;
	const float3 e1 = cameraPos - edgePoint2;
	const float3 e2 = edgePoint2 - edgePoint1;
	const float3 n = normalize(cross(e1, e0));
	float3 toIntersection, ratio3, lineIntersect;
	if (1)
	{
		toIntersection = directionN * dot(cameraPos - origin, n) / dot(n, directionN);
		const float3 intersectPos = origin + toIntersection;
		const float3 lineNormal2 = normalize(cross(normalize(e2), n));
		const float3 tolineIntersectPoint = (intersectPos - cameraPos)* dot(edgePoint1 - cameraPos, lineNormal2) / dot(intersectPos - cameraPos, lineNormal2);
		lineIntersect = cameraPos + tolineIntersectPoint;
		isOnrTiangle = false;
		*pIntersectWorld3 = intersectPos;
	}
	else
	{
		const float3 lineNormal2 = normalize(cross(normalize(e2), n));
		const float3 tolineIntersectPoint = directionN* dot(edgePoint1 - origin, lineNormal2) / dot(directionN, lineNormal2);
		//printf("tolineIntersectPoint:(%f,%f,%f)\n", tolineIntersectPoint.x, tolineIntersectPoint.y, tolineIntersectPoint.z);
		lineIntersect = origin + tolineIntersectPoint;
		isOnrTiangle = true;
		*pIntersectWorld3 = lineIntersect;

	}
	my_printf("intersect:(%f,%f,%f)\n", lineIntersect.x, lineIntersect.y, lineIntersect.z);

	*pLineIntersect = lineIntersect;
	ratio3 = (lineIntersect - edgePoint1) / (edgePoint2 - edgePoint1);

	my_printf("edge1:%f,%f,%f, edge2:%f,%f,%f\n", edgePoint1.x, edgePoint1.y, edgePoint1.z, edgePoint2.x, edgePoint2.y, edgePoint2.z);
	proj_ratio = getRatioInSpan(edgePoint1, edgePoint2, modelView_float, lineIntersect);
	//printf("linar(%f,%f,%f),rePorj_value:%f\n", ratio3.x, ratio3.y, ratio3.z, proj_ratio);

	//printf("proj_ratio:%f\n", proj_ratio);
	reversePoint3 = projective_interpo(edgePoint1, edgePoint2, d_modelViewRight, proj_ratio, 1);
	//printf("ratio3:%f,%f,%f\n", ratio3.x, ratio3.y, ratio3.z);
	//printf("dotValue:%f\n", dot(n, directionN));
	float lgap = 0.5 / noteSpan;
	//printf("noteSpane:%f\n", noteSpan);
	float lmin = 0 - lgap;
	float lmax = 1 + lgap;
	//printf("lmin,lmax:%f,%f\n", lmin, lmax);
	if (lmin < ratio3.x && ratio3.x <= lmax && lmin < ratio3.y && ratio3.y <= lmax && lmin < ratio3.z && ratio3.z <= lmax)
	{
		//printf("touched\n");
		return TRUE;
	}

	//printf("no touched\n");
	return FALSE;

}
__device__ int2 nearestTc(float2 tc)
{
	return make_int2(tc.x, tc.y);//直接进行int转换，因为减去0.5+0.5
}
__device__ int2 neighborTc(float2 tc)
{
	if (tc.x - floor(tc.x) > 0.5)
		return make_int2(tc.x + 1, tc.y);
	else
		return make_int2(tc.x - 1, tc.y);
}
//计算
__device__ float linarRatio(float x)
{
	float fracPart = x - floor(x);
	if (fracPart > 0.5)
		return (fracPart - 0.5);
	else
		return (0.5 - fracPart);

}
//映射到扭曲空间
__device__ float4 colorTextreNorTc(float2 tc)
{
	float2 nonNorTc = tc* make_float2(d_imageWidth, d_imageHeight);
	int2 mapTx = nearestTc(nonNorTc);
	int index = mapTx.y * d_imageWidth + mapTx.x;
	int mappedX = (int)(d_map_buffer[index].x + 0.5);

	//printf("mapped tc:(%d,%f),z:(%f)\n", mappedX, nonNorTc.y, tex2D(optixColorTex, nonNorTc.x, nonNorTc.y).z);
	return tex2D(optixColorTex, mappedX, nonNorTc.y);
}
__device__ int getNoteIndex(float2 tc)
{
	//printf("in note id\n");
	float2 nonNorTc = tc* make_float2(d_imageWidth, d_imageHeight);

	int2 mapTx = nearestTc(nonNorTc);
	//printf("mapped tc:(%d,%d)\n", mapTx.x, mapTx.y);
	int index = mapTx.y * d_imageWidth + mapTx.x;
	int noteId = (int)(d_map_buffer[index].z + 0.5);
	//printf("noteId tc:(%d)\n", noteId);
	return noteId;
}

__device__ bool isOccluedeArea(float2 tc)
{
	float2 nonNorTc = tc* make_float2(d_imageWidth, d_imageHeight);
	int2 mapTx = nearestTc(nonNorTc);
	int index = mapTx.y * d_imageWidth + mapTx.x;
	int mappedY = (int)(d_map_buffer[index].y + 0.5);
	if (mappedY < 1)
	{
		//printf("not occluded\n");
		return false;
	}
	//printf("occluded mapped coord:(%d,%d)\n", mappedY, mapTx.y);
	return true;

}
__device__ bool canGetMappedPosition(float2 tc, float4* poc)
{
	float2 nonNorTc = tc* make_float2(d_imageWidth, d_imageHeight);
	int2 mapTx = nearestTc(nonNorTc);
	int index = mapTx.y * d_imageWidth + mapTx.x;
	int mappedX = (int)(d_map_buffer[index].y + 0.5);
	if (mappedX < 1)
	{
		return false;
	}
	//printf("mapped coord:(%d,%d)\n", mappedY, mapTx.y);
	*poc = tex2D(optixColorTex, mappedX, nonNorTc.y);
	return true;

}
__device__ int noMappedPosition(float2 tc, int &mappedX,float4& poc)// 这个是坐标点对于
{
	float2 nonNorTc = tc;
	int2 mapTx = nearestTc(nonNorTc);
	int index = mapTx.y * d_imageWidth + mapTx.x;
    mappedX = (int)(d_map_buffer[index].y + 0.5);
	my_printf("mapped coord:(%d,%d)\n", mappedX, mapTx.y);
	poc = tex2D(optixColorTex, mappedX, nonNorTc.y);
	my_printf("color:(%f,%f,%f,%f)\n", poc.x, poc.y, poc.z, poc.w);
	if (poc.w >1500)// 在raytracing.cu里面定义了如果是冗余区域，返回99999
	{
		return RAYBACKTOMAIN;
	}
	if (mappedX < 1)
	{
		return RAYOUT;
	}
	return RAYVALID;

}
__device__ int linearNoMappedPosition(float2 tc, int eocWidth, EOCPixel & currentPixel, float4& oc)
{
	float2 nonNorTc = tc;
	float4 oc1,oc2;
	my_printf("tc:(%f,%f)\n", tc.x, tc.y);
	bool centerisAtEdge,neighberIsAtEdge;
	{
		int2 mapTx = nearestTc(nonNorTc);
	
		int index = mapTx.y * d_imageWidth + mapTx.x;
		int mappedX = (int)(d_map_buffer[index].y + 0.5);
		 oc1 = tex2D(optixColorTex, mappedX, nonNorTc.y);
		 my_printf("mappedX:%d\n", mappedX);

		 if (mappedX < 1)
		 {
			 return RAYOUT;
		 }
		 centerisAtEdge = d_edge_buffer[mapTx.y*eocWidth + mappedX].isRightEdge;
		
	}
	{
		int2 neighborTx = neighborTc(nonNorTc);
		//my_printf("neighborTx:(%d,%d)\n", neighborTx.x, neighborTx.y);

		int neighborindex = neighborTx.y * d_imageWidth + neighborTx.x;
	

		int neighborMappedX = (int)(d_map_buffer[neighborindex].y + 0.5);
		my_printf("neighborMappedX:%d\n", neighborMappedX);

		oc2 = tex2D(optixColorTex, neighborMappedX, nonNorTc.y);
		if (neighborMappedX < 1)
		{
			return RAYOUT;
		}
		neighberIsAtEdge = d_edge_buffer[neighborTx.y*eocWidth + neighborindex].isRightEdge;
	}
	if (centerisAtEdge || neighberIsAtEdge)
	{
		currentPixel.m_isValid = false;
		return RAYOUT;
	}
	else
		currentPixel.m_isValid = true;
	if (oc1.w > 1500 || oc2.w > 1500)// 在raytracing.cu里面定义了如果是冗余区域，返回99999
	{
		return RAYBACKTOMAIN;
	}
	my_printf("oc1:(%f,%f,%f,%f)\n", oc1.x, oc1.y, oc1.z, oc1.w);
	my_printf("oc2:(%f,%f,%f,%f)\n", oc2.x, oc2.y, oc2.z, oc2.w);

	float ratio = linarRatio(tc.x);
	float ratioZ = repo(repo(oc1.w) + ratio*(repo(oc2.w) - repo(oc1.w)));
	float zRatio = (ratioZ - oc1.w) / (oc2.w - oc1.w);
	my_printf("ratioZ,zRatio:(%f,%f)\n", ratioZ, zRatio);

	oc = oc1 +  (oc2 - oc1)*zRatio;
	my_printf("interpolated color:(%f,%f,%f,%f)\n", oc.x, oc.y, oc.z, oc.w);

	return RAYVALID;
	
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
__device__ float2 getCameraTc(float3 pos, float* modelMat, float* projMat)
{
	float4 temp = MutiMatrixN(modelMat, make_float4(pos, 1));
	temp = MutiMatrixN(projMat, temp);
	temp = temp / temp.w;
	float2 tc;
	tc.x = 0.5*temp.x + 0.5;
	tc.y = 0.5*temp.y + 0.5;
	return tc;
}

class List
{

};

__device__ float getRatioInSpan(float3 beginPos, float3 endPos, float* p_modelView, float3 testPos)
{
	float x, y, z1, z2, z3;
	float4 temp = MutiMatrixN(p_modelView, make_float4(beginPos.x, beginPos.y, beginPos.z, 1));
	z1 = temp.z;
	temp = MutiMatrixN(p_modelView, make_float4(endPos.x, endPos.y, endPos.z, 1));
	z2 = temp.z;
	temp = MutiMatrixN(p_modelView, make_float4(testPos.x, testPos.y, testPos.z, 1));
	z3 = temp.z;
	float real_ratio = (repo(z1) - repo(z3)) / (repo(z1) - repo(z2));
	/*
	printf("in projection\n");
	printf("beginPos:(%f,%f,%f)\n", beginPos.x, beginPos.y, beginPos.z);
	printf("endPos:(%f,%f,%f)\n", endPos.x, endPos.y, endPos.z);
	printf("testPos:(%f,%f,%f)\n", testPos.x, testPos.y, testPos.z);
	printf("z1:%f,z2:%f,z3:%f\n",z1,z2,z3);
	printf("repo z1:%f,z2:%f,z3:%f\n", repo(z1), repo(z2), repo(z3));
	*/
	//printf("ration:%f\n", real_ratio);

	return real_ratio;
	return (z3 - z2) / (z1 - z2);
}
__device__ float3 projective_interpo(float3 beginPos, float3 endPos, float* p_modelView, float ratio, int print = 0)
{
	float x, y, z1, z2;
	float4 temp = MutiMatrixN(p_modelView, make_float4(beginPos, 1));
	z1 = temp.z;
	temp = MutiMatrixN(p_modelView, make_float4(endPos, 1));
	z2 = temp.z;
	float real_z = repo((1 - ratio) *repo(z1) + ratio* repo(z2));
	float real_ratio = (real_z - z1) / (z2 - z1);
	if (print)
	{
		//	 printf("z1:%f,z2:%f,ratio:%f,real_z:%f,real_ratio:%f\n",z1,z2,ratio, real_z,real_ratio);

	}
	return beginPos*(1 - real_ratio) + endPos  *  real_ratio;
}
__device__ bool isVolume(float2 uv)
{
	float4 value = tex2D(cudaOccuderTex, uv.x, uv.y);
	return value.x > 0.5;
}
__device__ bool isVolumeTop(float2 uv)
{
	float4 value = tex2D(cudaTopOccuderTex, uv.x, uv.y);
	return value.x > 0.5;

}
__device__ bool isEdge(float2 uv)
{
	return tex2D(cudaEdgeTex, uv.x, uv.y).x > 0.05;
}
__device__ bool isEdgeTop(float2 uv)
{
	return tex2D(cudaEdgeTex, uv.x, uv.y).y > 0.05;
}
__device__ bool isMinusEdge(float2 uv)
{
	return tex2D(cudaEdgeTex, uv.x, uv.y).x < -0.05;
}
__device__ bool isMinusEdgeTop(float2 uv)
{
	return tex2D(cudaEdgeTex, uv.x, uv.y).y < -0.05;
}
__device__ bool isTracingEdge(float2 tc)
{
	float2 nonNorTc = tc* make_float2(d_imageWidth, d_imageHeight);
	return isEdge(nonNorTc) || isEdgeTop(nonNorTc) || isMinusEdge(nonNorTc) || isMinusEdgeTop(nonNorTc);
}
__device__ bool isRightEdge(float2 tc)
{
	float2 nonNorTc = tc* make_float2(d_imageWidth, d_imageHeight);
	return  isEdge(nonNorTc);
}
__device__ bool isTopEdge(float2 tc)
{
	float2 nonNorTc = tc* make_float2(d_imageWidth, d_imageHeight);
	return  isEdgeTop(nonNorTc);
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
	//if (index != 290)
		//return;
	int arrayNum = index;
	int accumNum = 0;
	int state = 0;
	pixelEnum etype = notVolumn;
	unsigned int* nextPtr = &d_cudaPboTopBuffer[arrayNum].x;
	int listIndex;
	int lastMinusIndey = 0;
	for (int y = 0; y< d_imageHeight; y++)
	{
		float2 currentUv = toUv(index, y);
		if (isMinusEdgeTop(currentUv))
		{
			lastMinusIndey = y;
		}
		if (isVolumeTop(currentUv) && etype == notVolumn)
		{
		//	printf("insert :%d\n", y);
			listIndex = atomicAdd(&d_atomic_top, 1);
			atomicExch(nextPtr, listIndex);// write listIndex to next slot
			d_listBuffer_top[listIndex].beginIndex = y;
			d_listBuffer_top[listIndex].endIndex = y;
			d_listBuffer_top[listIndex].nextPt = 0;
			d_listBuffer_top[listIndex].leftEdge = lastMinusIndey;
			nextPtr = (unsigned int *)(&(d_listBuffer_top[listIndex].nextPt));

			etype = isVolumn;
		}
		else if (isVolumeTop(currentUv) && etype == isVolumn)
		{


		}
		else if (etype == isVolumn && isEdgeTop(currentUv))
		{
			//printf("end :%d\n",y);


			d_listBuffer_top[listIndex].endIndex = y - 1;
			etype = notVolumn;
		}
	}
}
// 记录interval  interval的leftEdge 记录最左边的edge（为非冗余渲染用），beginIndex记录第一个遮挡的像素。endIndex 记录的是右边界（右边边界像素-1位）
// 每一个
__global__ void countRowKernel(int kernelWidth, int kernelHeight)
{
	int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (y > kernelHeight)
		return;
	//if (y != 838)
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
		if (isVolume(currentUv) && etype == notVolumn)
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
		else if (isVolume(currentUv) && etype == isVolumn)
		{


		}
		else if (etype == isVolumn && isEdge(currentUv))
		{
			//printf("end :%d\n", x);

			d_listBuffer[listIndex].endIndex = x - 1;
			//d_listBuffer[listIndex].endIndex = d_listBuffer[listIndex].beginIndex + 117;
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

// 在第y 行的beginX 到endX直接记录中空的区域的位置信息leftEdge是用来评估左边边界，endUv是用来算左右两边的深度插值
__device__ void FillVolumn(int beginX, int endX, int y, int endUv, int leftEdge, int accumIndex, int  noteIndex)
{
	int top = min(endX, d_outTextureWidth - 1);
	//printf("volumn begin:%d,end:%d,top:%d\n",beginX,endX,top);
	float2 beforeEdgeUv = toUv(endUv - 1, y);
	float3 beforePos = make_float3(tex2D(cudaPosTex, beforeEdgeUv.x, beforeEdgeUv.y));
	float2 endEdgeUv = toUv(endUv, y);
	float3 endPos = make_float3(tex2D(cudaPosTex, endEdgeUv.x, endEdgeUv.y));
	// 记录高点
	float2 leftEdgeUv = toUv(leftEdge + 1, y);
	float3 leftEdgePos = make_float3(tex2D(cudaPosTex, leftEdgeUv.x, leftEdgeUv.y));
	float3 eoc_pos = d_eocPos;
	/*printf("endPos:(%f,%f,%f)\n", endPos.x, endPos.y, endPos.z);
	printf("beforePos:(%f,%f,%f)\n", beforePos.x, beforePos.y, beforePos.z);
	printf("leftEdgePos:(%f,%f,%f)\n", leftEdgePos.x, leftEdgePos.y, leftEdgePos.z);
	printf("camera:(%f,%f,%f)\n", d_cameraPos.x, d_cameraPos.y, d_cameraPos.z);
	printf("eoc:(%f,%f,%f)\n", eoc_pos.x, eoc_pos.y, eoc_pos.z);*/

	//for (int i = 0; i < 4; i++)
	//	printf("(%f,%f,%f,%f)\n", d_modelView[4 * i + 0], d_modelView[4 * i + 1], d_modelView[4 * i + 2], d_modelView[4 * i + 3]);
	for (int x = beginX; x <= top; x++)
	{
		int lenght = (top + 1 - beginX);
		float ratio = (x * 1.0f - beginX*1.0f) / lenght;
		float3 realPos = projective_interpo(beforePos, endPos, d_modelViewRight, ratio);
		int index = y*d_outTextureWidth + x;
		float dis = distance(leftEdgePos, realPos, eoc_pos);
		d_cudaTexture[index] = make_float4(dis, realPos.x, realPos.y, realPos.z);
		//printf("x:%d,realpos(%f,%f,%f)\n", x, realPos.x, realPos.y, realPos.z);
		int originMappos = y*d_imageWidth + accumIndex - lenght;
		d_map_buffer[originMappos].y = x;
		d_map_buffer[originMappos].z = noteIndex;
		accumIndex++;
	}
}

__device__ void FillVolumnTop(int beginY, int endY, int x, int endUv, int leftEdge, int accumIndex, int  noteIndex)
{
	int top = min(endY, d_outTopTextureHeight);
	//printf("volumn begin:%d,end:%d,top:%d,endUv:%d\n", beginY, endY, top, endUv);
	float2 beforeEdgeUv = toUv(x, endUv - 1);
	float3 beforePos = make_float3(tex2D(cudaPosTex, beforeEdgeUv.x, beforeEdgeUv.y));
	float2 endEdgeUv = toUv(x, endUv);
	float3 endPos = make_float3(tex2D(cudaPosTex, endEdgeUv.x, endEdgeUv.y));
	// 记录高点
	float2 leftEdgeUv = toUv(x, leftEdge + 1);
	float3 leftEdgePos = make_float3(tex2D(cudaPosTex, leftEdgeUv.x, leftEdgeUv.y));
	float3 eoc_pos = d_eocTopPos;
	/*printf("endPos:(%f,%f,%f)\n", endPos.x, endPos.y, endPos.z);
	printf("beforePos:(%f,%f,%f)\n", beforePos.x, beforePos.y, beforePos.z);
	printf("leftEdgePos:(%f,%f,%f)\n", leftEdgePos.x, leftEdgePos.y, leftEdgePos.z);
	printf("camera:(%f,%f,%f)\n", d_cameraPos.x, d_cameraPos.y, d_cameraPos.z);
	printf("eoc:(%f,%f,%f)\n", eoc_pos.x, eoc_pos.y, eoc_pos.z);*/

	//for (int i = 0; i < 4; i++)
	//	printf("(%f,%f,%f,%f)\n", d_modelView[4 * i + 0], d_modelView[4 * i + 1], d_modelView[4 * i + 2], d_modelView[4 * i + 3]);
	for (int y = beginY; y < top; y++)
	{
		int lenght = (top + 1 - beginY);
		float ratio = (y * 1.0f - beginY*1.0f) / lenght;
		float3 realPos = projective_interpo(beforePos, endPos, d_modelViewTop, ratio);
		int index = y*d_outTopTextureWidth + x;
		float dis = distance(leftEdgePos, realPos, eoc_pos);
		d_cudaTopTexture[index] = make_float4(-dis, realPos.x, realPos.y, realPos.z);
		//printf("y:%d,realpos(%f,%f,%f)\n", y, realPos.x, realPos.y, realPos.z);
		int originMappos = y*d_imageWidth + accumIndex - lenght;
		d_map_buffer[originMappos].y = y;
		d_map_buffer[originMappos].z = noteIndex;
		accumIndex++;
	}
}
__device__ void FillVolumnTop(int beginY, int endY, int x, int endUv, int leftEdge)
{
	int top = min(endY, d_outTopTextureHeight);
	//printf("volumn begin:%d,end:%d,top:%d\n",beginX,endX,top);
	float3 beforePos = make_float3(tex2D(cudaPosTex, x, endUv - 1));
	//printf("beforePos(%d,%d)", x, endUv - 1);
	float3 endPos = make_float3(tex2D(cudaPosTex, x, endUv));
	//printf("endPos(%d,%d)", x, endUv );
	float3 leftEdgePos = make_float3(tex2D(cudaPosTex, x, leftEdge + 1));
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
		float3 realPos = projective_interpo(beforePos, endPos, d_modelView, ratio);
		int index = y*d_outTopTextureWidth + x;
		float dis = distance(leftEdgePos, realPos, ecoCamera);
		//printf("y:%d,realpos(%f,%f,%f)\n", y, realPos.x, realPos.y, realPos.z);

		d_cudaTopTexture[index] = make_float4(-dis, realPos.x, realPos.y, realPos.z);
	}
}
__device__ void FillSpanTop(int beginY, int endY, int x, float2 beginUv, float2 endUv)
{
	int top = min(endY, d_outTopTextureHeight);
	//printf("fill from %d to %d at line %d", beginY, endY, x);
	//printf("begin(%f,%f),end(%f,%f),d_outTextureWidth:%d\n", beginUv.x, beginUv.y, endUv.x, endUv.y, d_outTextureWidth);
	//printf("endY:%d,d_outTopTextureHeight:%d,top:%d\n", endY, d_outTopTextureHeight, top);

	for (int y = beginY; y <= top; y++)
	{
		int index = y*d_outTopTextureWidth + x;
		float uvy = beginUv.y + (endUv.y - beginUv.y)*(y - beginY) / (top - beginY);
		d_cudaTopTexture[index] = FillPoint(beginUv.x - 0.5, uvy - 0.5);//tex2D(cudaColorTex, beginUv.x, uvy);
	}
}
//beginUv 记录的是 toUv(index, texBegin), toUv(index, texEnd) 整数
__device__ void FillSpanTop(int beginY, int endY, int x, float2 beginUv, float2 endUv, int* accumIndexX) //beginUv 用了toUv函数
{
	int top = min(endY, d_outTextureWidth - 1);
	for (int y = beginY; y <= top; y++)
	{
		int index = y*d_outTopTextureWidth + x;
		float uvy = beginUv.y + (endUv.y - beginUv.y)*(y - beginY) / (top - beginY);
		d_cudaTopTexture[index] = FillPoint(beginUv.x , uvy );//tex2D(cudaColorTex, beginUv.x, uvy);

		//printf("write uv(%f,%f)\n",uvx,beginUv.y);
		//记录映射关系
		int originMappos = y*d_imageWidth + *accumIndexX;
		//printf("y:%d,mappedPos:%d\n", y, *accumIndexX);
		d_map_buffer[originMappos].x = y;
		*accumIndexX += 1;
	}
}

// 在第y 行的beginX 到endX直接记录中空的区域的位置信息leftEdge是用来评估左边边界，endUv是用来算左右两边的深度插值

// 在第y 行的beginX 到endX闭区间 的这一块区域插入原来图像从beginUV到endUV的这一横条图像accumIndexX 记录原来图像到新图像的映射
__device__ void FillSpan(int beginX, int endX, int y, float2 beginUv, float2 endUv, int* accumIndexX) //beginUv 用了toUv函数
{
	int top = min(endX, d_outTextureWidth - 1);
	for (int x = beginX; x <= top; x++)
	{
		int index = y*d_outTextureWidth + x;
		float uvx = beginUv.x + (endUv.x - beginUv.x)*(x - beginX) / (top - beginX);
		d_cudaTexture[index] = tex2D(cudaColorTex, uvx, beginUv.y);
		//printf("write uv(%f,%f)\n",uvx,beginUv.y);
		//记录映射关系
		int originMappos = y*d_imageWidth + *accumIndexX;
		//printf("x:%d,mappedPos:%d\n", x, *accumIndexX);
		d_map_buffer[originMappos].x = x;
		*accumIndexX += 1;
	}
}
__global__ void renderToTexutre(int kernelWidth, int kernelHeight)
// 竖需要改动
{
	int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (y > kernelHeight)
		return;
	//if (y != 813)
	//	return;
	int listIndex = y;
	int rowLength = d_imageWidth;
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
	ListNote currentNote = *((ListNote*)&d_cudaPboBuffer[listIndex]);
	int leftEdgeIndex = 0;

	while (currentNote.nextPt != 0)
	{
		/*if (listIndex == 836)
		{
		ListNote currentNote2 = *((ListNote*)&d_cudaPboBuffer[listIndex]);
		printf("Render:next %d end:%d begin:%d,\n", currentNote2.nextPt, currentNote2.endIndex, currentNote2.beginIndex);
		}*/
		int noteIndex = currentNote.nextPt;
		currentNote = d_listBuffer[currentNote.nextPt];
		//printf("current:b:%d,e:%d,n:%d,leftEdge:%d\n", currentNote.beginIndex, currentNote.endIndex, currentNote.nextPt, currentNote.leftEdge);
		texEnd = currentNote.endIndex;
		span = currentNote.endIndex - currentNote.beginIndex + 1;
		leftEdgeIndex = currentNote.leftEdge;
		fillBegin = texBegin + acuumPixel;
		fillEnd = texEnd + acuumPixel;
		FillSpan(fillBegin*factor, fillEnd*factor, y, toUv(texBegin, y), toUv(texEnd, y), &accum_index);  //内层 for 循环，左闭右闭
		FillVolumn((fillEnd + 1)*factor, (fillEnd + span)*factor, y, texEnd + 1, leftEdgeIndex, accum_index, noteIndex);//内层 for 循环，左闭右闭
		acuumPixel += span;
		texBegin = currentNote.endIndex + 1;
	}
	fillBegin = texBegin + acuumPixel;
	//printf("final:(%d,%d) u(%f,%f)\n", fillBegin, d_imageWidth + span, toUv(texBegin, y).x, toUv(d_imageWidth - 1, y).x);
	FillSpan(fillBegin*factor, (d_imageWidth + acuumPixel)*factor, y, toUv(texBegin, y), toUv(d_imageWidth - 1, y), &accum_index);


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
	//if (index != 305)
	//	return;
	int listIndex = index;
	int rowLength = d_imageWidth;
	ListNote currentNote = *((ListNote*)&d_cudaPboTopBuffer[listIndex]);
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
	int accum_index = 0;  // 记录累计
	float factor = d_imageWidth*1.0 / rowLength;
	currentNote = *((ListNote*)&d_cudaPboTopBuffer[listIndex]);
	int leftEdgeIndex = 0;
	while (currentNote.nextPt != 0)
	{

		int noteIndex = currentNote.nextPt;
		currentNote = d_listBuffer_top[currentNote.nextPt];
		//printf("current:b:%d,e:%d,n:%d,leftEdge:%d\n", currentNote.beginIndex, currentNote.endIndex, currentNote.nextPt, currentNote.leftEdge);

		texEnd = currentNote.endIndex;
		span = currentNote.endIndex - currentNote.beginIndex + 1;
		leftEdgeIndex = currentNote.leftEdge;
		fillBegin = texBegin + acuumPixel;
		fillEnd = texEnd + acuumPixel;
		//printf("accum_index:%d,accum_index:%d,n:%d\n", accum_index, accum_index);
		//FillSpanTop(fillBegin*factor, fillEnd*factor, index, toUv(index, texBegin), toUv(index, texEnd));  //for 循环，左闭右开
		//FillVolumnTop((fillEnd)*factor, (fillEnd + span)*factor, index, texEnd, leftEdgeIndex);
		FillSpanTop(fillBegin*factor, fillEnd*factor, index, toUv(index, texBegin), toUv(index, texEnd), &accum_index);  //内层 for 循环，左闭右闭
		//printf("accum_index:%d\n", accum_index);

		FillVolumnTop((fillEnd + 1)*factor, (fillEnd + span)*factor, index, texEnd + 1, leftEdgeIndex, accum_index, noteIndex);//内层 for 循环，左闭右闭
	
		

		acuumPixel += span;
		texBegin = currentNote.endIndex+1;

		//printf("texBegin:%d,acuumPixel:%d,n:%d\n", texBegin, acuumPixel);

	}
	fillBegin = texBegin + acuumPixel;
	//printf("final:(%d,%d) u(%f,%f)\n", fillBegin, d_imageWidth + span, toUv(index, texBegin).y, toUv(index,d_imageWidth).y);
	//FillSpanTop(fillBegin*factor, (d_imageWidth + acuumPixel)*factor, index, toUv(index, texBegin), toUv(index, d_imageWidth - 1));
	FillSpanTop(fillBegin*factor, (d_imageWidth + acuumPixel)*factor, index, toUv(index, texBegin), toUv(index, d_imageWidth - 1), &accum_index);
	

}
ListNote *device_data, *device_top_data = NULL;
int atomBuffer = 1; // 原子计数从1开始，0作为空节点标识位
int atomBufferTop = 1;
#ifdef DEBUG
ListNote *host_data = NULL;
#endif
extern void cudaInit(int height, int width, int k, float rowLarger)
{
	checkCudaErrors(cudaMalloc(&device_data, height*k*sizeof(ListNote)));
	checkCudaErrors(cudaMalloc(&device_top_data, ROWLARGER*width*k*sizeof(ListNote)));

	checkCudaErrors(cudaMemcpyToSymbol(d_listBuffer, &device_data, sizeof(ListNote*)));
	checkCudaErrors(cudaMemcpyToSymbol(d_listBuffer_top, &device_top_data, sizeof(ListNote*)));


	checkCudaErrors(cudaMemcpyToSymbol(d_atomic, &atomBuffer, sizeof(int)));

	checkCudaErrors(cudaMemcpyToSymbol(d_atomic_top, &atomBufferTop, sizeof(int)));
	checkCudaErrors(cudaMemset(device_data, 0, height*k*sizeof(ListNote)));
	//checkCudaErrors(cudaMemset(cuda_TexturePbo_buffer, 0, width* height*rowLarger*sizeof(float4)));
#ifdef DEBUG
	checkCudaErrors(cudaMallocHost(&host_data, height*k*sizeof(ListNote)));
#endif
	checkCudaErrors(cudaMalloc(&modelView, 16 * sizeof(float)));
	checkCudaErrors(cudaMalloc(&modelView_inv, 16 * sizeof(float)));

	checkCudaErrors(cudaMalloc(&proj, 16 * sizeof(float)));
	checkCudaErrors(cudaMalloc(&proj_inv, 16 * sizeof(float)));

	checkCudaErrors(cudaMalloc(&modelViewRight, 16 * sizeof(float)));
	checkCudaErrors(cudaMalloc(&modelViewTop, 16 * sizeof(float)));

	checkCudaErrors(cudaMalloc(&cuda_map_buffer, width*height * sizeof(float4)));
	checkCudaErrors(cudaMemcpyToSymbol(d_map_buffer, &cuda_map_buffer, sizeof(float*)));

	//host_data = (ListNote*)malloc(height*k*sizeof(ListNote));
	//memset(host_data, 0, height*k*sizeof(ListNote));
	//checkCudaErrors(cudaMemcpy((void *)device_data, (void *)host_data, height * k * sizeof(ListNote), cudaMemcpyDeviceToHost));

	const int eocWidth = width*rowLarger;
	const int eocHeight = height*rowLarger;
	checkCudaErrors(cudaMalloc(&edgeBuffer, eocWidth*eocHeight*sizeof(edgeInfo)));
	checkCudaErrors(cudaMemcpyToSymbol(d_edge_buffer, &edgeBuffer, sizeof(edgeInfo*)));


}
extern "C" void countRow(int width, int height, Camera * pCamera, Camera * pEocCam, Camera * pEocTopCamera)
{
	checkCudaErrors(cudaMemset(cuda_TexturePbo_buffer, 0, ROWLARGER*width*height*sizeof(float4)));
	checkCudaErrors(cudaMemset(cuda_top_TexturePbo_buffer, 0, ROWLARGER*width*ROWLARGER*height*sizeof(float4)));
	checkCudaErrors(cudaMemset(cuda_map_buffer, 0, width*height*sizeof(float4)));

	checkCudaErrors(cudaMemcpyToSymbol(d_cameraPos, &pCamera->getCameraPos(), 3 * sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(d_eocPos, &pEocCam->getCameraPos(), 3 * sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(d_eocTopPos, &pEocTopCamera->getCameraPos(), 3 * sizeof(float)));

	checkCudaErrors(cudaMemcpyToSymbol(d_rightRD, &pEocCam->getDirectionR(), 3 * sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(d_topRD, &pEocTopCamera->getDirectionR(), 3 * sizeof(float)));

	nv::matrix4f mod_mat = nv::matrix4f(pCamera->getModelViewMat());
	nv::matrix4f mod_inv_mat = inverse(mod_mat);
	checkCudaErrors(cudaMemcpy(modelView, mod_mat.get_value(), 16 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_modelView, &modelView, sizeof(float*)));
	checkCudaErrors(cudaMemcpy(modelView_inv, mod_inv_mat.get_value(), 16 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_modelView_inv, &modelView_inv, sizeof(float*)));

	nv::matrix4f proj_mat = nv::matrix4f(pCamera->getProjection());
	nv::matrix4f proj_inv_mat = inverse(proj_mat);
	checkCudaErrors(cudaMemcpy(proj, proj_mat.get_value(), 16 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_proj, &proj, sizeof(float*)));
	checkCudaErrors(cudaMemcpy(proj_inv, proj_inv_mat.get_value(), 16 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_proj_inv, &proj_inv, sizeof(float*)));


	checkCudaErrors(cudaMemcpy(modelViewRight, pEocCam->getModelViewMat(), 16 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_modelViewRight, &modelViewRight, sizeof(float*)));


	checkCudaErrors(cudaMemcpy(modelViewTop, pEocTopCamera->getModelViewMat(), 16 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_modelViewTop, &modelViewTop, sizeof(float*)));


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
	checkCudaErrors(cudaMemcpyToSymbol(d_atomic_top, &atomBufferTop, sizeof(int)));
	checkCudaErrors(cudaMemset(cuda_PBO_Top_Buffer, 0, ROWLARGER*height*sizeof(ListNote)));

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
		checkCudaErrors(cudaBindTextureToArray(cudaEdgeTex, tmpcudaArray, channelDesc));
		cudaEdgeTex.filterMode = cudaFilterModePoint;
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
		cudaPosTex.filterMode = cudaFilterModePoint; // 最后还是改成手工interpolate

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
	else if (posBlend_t == pResouce->getType())
	{
		checkCudaErrors(cudaBindTextureToArray(posBlendTex, tmpcudaArray, channelDesc));
		posBlendTex.filterMode = cudaFilterModeLinear;
	}
	else if (normal_t == pResouce->getType())
	{

		checkCudaErrors(cudaBindTextureToArray(cudaNormalTex, tmpcudaArray, channelDesc));
		cudaNormalTex.filterMode = cudaFilterModePoint;
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
	else if (list_top_e == pResource->getType())
	{
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&cuda_PBO_Top_Buffer, &numBytes, *pCudaTex));
		checkCudaErrors(cudaMemcpyToSymbol(d_cudaPboTopBuffer, &cuda_PBO_Top_Buffer, sizeof(ListNote*)));
	}
	else if (to_optix_t == pResource->getType())
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
	checkCudaErrors(cudaMemcpy(modelView_construct_inv, invModelView.get_value(), 16 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_modeView_inv_construct, &modelView_construct_inv, sizeof(float*)));

	nv::vec2f bbmin = nv::vec2f(pReconstructCamer->getImageMin().x, pReconstructCamer->getImageMin().y);
	nv::vec2f bbmax = nv::vec2f(pReconstructCamer->getImageMax().x, pReconstructCamer->getImageMax().y);
	checkCudaErrors(cudaMemcpyToSymbol(d_bbmin, &bbmin, 2 * sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(d_bbmax, &bbmax, 2 * sizeof(float)));
	nv::vec2f tc = nv::vec2f(412, 512) / nv::vec2f(1024.0, 1024.0);
	nv::vec2f xy = bbmin + (bbmax - bbmin)*tc;
	nv::vec4f temp;
	MutiMatrix((float*)nv::vec4f(xy, -1, 1), (float*)invModelView.get_value(), (float*)& temp);// *;
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

__device__ float3 getImagePos(float2 tc, float* modelViewInv)
{
	float2 xy = d_bbmin + (d_bbmax - d_bbmin)*tc;
	xy = xy;
	float4 temp = MutiMatrixN(modelViewInv, make_float4(xy.x, xy.y, -1, 1));// must
	temp = temp / temp.w;
	return make_float3(temp.x, temp.y, temp.z);
}
__device__ float3 toFloat3(float4 inValue)
{
	return make_float3(inValue.x / inValue.w, inValue.y / inValue.w, inValue.z / inValue.w);
}


__device__ EOCPixel occludedRayLinarState(float tex, int yIndex, float texBegin, float span, float enterZ, float dpdx, float enter_projRatio, float4& color)
{
	EOCPixel currentPixel;
	float ratioOneD = (tex - texBegin) / span;
	float currentZ = repo(repo(enterZ) + (ratioOneD - enter_projRatio)*dpdx);
	const int eocWidth = d_imageWidth*ROWLARGER;

	int texFetchState = linearNoMappedPosition(make_float2(tex, yIndex + 0.5), eocWidth, currentPixel, color);
	if (RAYVALID == texFetchState)
	{
		float zOnTex = color.w;
		currentPixel.m_detpth = zOnTex;
		currentPixel.m_isValid = true;
		my_printf("zOnTex:%f,currentZ:%f\n", zOnTex, currentZ);
		if (currentZ > zOnTex)
		{
			currentPixel.m_state = RAYISUNDER;
		}
		else
		{
			currentPixel.m_state = RAYISUP;
		}
	}
	else
	{
		currentPixel.m_isValid = false;
	}
	return currentPixel;

}

//#define RAYISUP 0
//#define RAYOUT 1
//#define RAYISUNDER 2
//如果光线比较远返回RAYISUNDER，如果比较近返回RAYISUP，如果不合理返回RAYOUT，如果返回主相机则是RAYBACKTOMAIN
//dpdx 记录的是光线在x方向上的
__device__ EOCPixel occludedRayState(float tex, int yIndex, float texBegin, float span, float enterZ, float dpdx, float enter_projRatio, float4& color)
{
	EOCPixel currentPixel;
	float ratioOneD = (tex - texBegin) / span;
	float currentZ = repo(repo(enterZ) + (ratioOneD - enter_projRatio)*dpdx);
	my_printf("ray Z:%f\n", currentZ);
	int mappedX;
	int texFetchState = noMappedPosition(make_float2(tex, yIndex + 0.5), mappedX, color);
	//这里的w指示到右相机视点的距Zdis,是正值，
	if (RAYVALID == texFetchState)
	{
		float zOnTex = color.w;
		currentPixel.m_detpth = zOnTex;
		currentPixel.m_isValid = true;
		//printf("mapped Z:%f\n", zOnTex);
		//如果光线比较远则距离值大
		//currentPixel.m_isAtEdge = abs(currentZ - zOnTex) > 3;
		const int eocWidth = d_imageWidth*ROWLARGER;
		currentPixel.m_isAtEdge = d_edge_buffer[yIndex*eocWidth+mappedX].isRightEdge;
		currentPixel.m_isValid = currentPixel.m_isValid&(!currentPixel.m_isAtEdge);
		float dzdx = d_edge_buffer[yIndex*eocWidth + mappedX].dzdx;

		if (currentZ > zOnTex)
		{
			currentPixel.m_state = RAYISUNDER;
		}
		else
		{
			currentPixel.m_state = RAYISUP;
		}
	}
	else
	{
		currentPixel.m_isValid = false;
	}
	return currentPixel;
}
__device__ float3 getIntersection(float3 pos, float3 normal, float3& camera, float3& lookAtPos)
{
	float3 toPlane = pos - camera;
	float3 toLookAt = lookAtPos - camera;
	float projToPlane = dot(toPlane, normal);
	float projToLookAt = dot(toLookAt, normal);
	float3 predict = camera + projToPlane*toLookAt / projToLookAt;
	return predict;
}

__device__ float3 plateInterpolation(float3 pos, float3 dir, float2 ndc, float3 cameraPos)
{
	float2 NDC = make_float2(ndc.x / d_imageWidth, ndc.y / d_imageHeight) * 2 - 1.0;
	float4 temp = MutiMatrixN(d_modelView_inv, MutiMatrixN(d_proj_inv, make_float4(NDC.x, NDC.y, 0.0f, 1.0f)));
	temp = temp / temp.w;
	float3 predictNear = make_float3(temp.x, temp.y, temp.z);
	float3 predictPos = getIntersection(pos, dir, cameraPos, predictNear);
	return predictPos;
}
__device__ bool pixelValid(int x, int y)
{
	float pocw = tex2D(optixColorTex, x + 0.5, y + 0.5).w;
	return pocw >0;
}
__device__ float mfracf(float x)
{
	return x - floorf(x);

}
__device__ int intersectCameraMidID(float3 posW, float3 directionW, float3 cameraPos, ListNote currentNote, int yIndex, bool isRayUp, int occludedObjId,
	float* modelView,// 查询modelView 相机下的深度
	bool& returnToMainC, EOCPixel &lastPixel, float4& intersectColor, float2& exitTC, float3& exitWorldPos)
{
	int texEnd = currentNote.endIndex;  // 这个是右边边界-的值
	int texBegin = currentNote.beginIndex;
	int span = texEnd + 1 - texBegin;
	int currentObjectId = nearestInt(tex2D(cudaNormalTex, texEnd - span / 2.0, yIndex + 0.5).w);
	if (currentObjectId != occludedObjId)
	{
		return OHTEROBJECT;
	}
	float dpdx;
	{
		float exitY, enterY, before = texEnd + 0.5, end = texEnd + 1 + 0.5;
#define GAP 0.01
		if (isRayUp)
		{
			exitY = yIndex + 1.0 - GAP;
			enterY = yIndex + GAP;
		}
		else
		{
			enterY = yIndex + 1.0 - GAP;
			exitY = yIndex + GAP;
		}
#undef GAP

		float2 beforeEnterTc = make_float2(before, enterY);                 //left
		float2 endEnterTc = make_float2(end, enterY);
		float2 beforeExitTc = make_float2(before, exitY);                 //left
		float2 endExitTc = make_float2(end, exitY);

		float3 beforeCenterPos = make_float3(tex2D(cudaPosTex, before, yIndex + 0.5));
		float3 beforeNormal = normalize(d_cameraPos - beforeCenterPos);
		float3 endCenterPos = make_float3(tex2D(cudaPosTex, end, yIndex + 0.5));
		float3 endNormal = normalize(d_cameraPos - endCenterPos);

		float3 beforeEnterPos = plateInterpolation(beforeCenterPos, beforeNormal, beforeEnterTc, d_cameraPos);
		float3 endEnterPos = plateInterpolation(endCenterPos, endNormal, endEnterTc, d_cameraPos);
		float3 beforeExitPos = plateInterpolation(beforeCenterPos, beforeNormal, beforeExitTc, d_cameraPos);
		float3 endExitPos = plateInterpolation(endCenterPos, beforeNormal, endExitTc, d_cameraPos);


		float enter_projRatio, exit_projRatiok;
		float3 enterReservedPos, exitReservedPos, _;
		bool f_;
		rayIntersertectTriangle(posW, normalize(directionW), cameraPos, beforeEnterPos, endEnterPos, d_modelViewRight, span, &enterReservedPos, &_, f_, enter_projRatio, _);
		rayIntersertectTriangle(posW, normalize(directionW), cameraPos, beforeExitPos, endExitPos, d_modelViewRight, span, &exitReservedPos, &_, f_, exit_projRatiok, _);

		float2 camera1Entertc = getCameraTc(enterReservedPos, d_modelViewRight, d_proj);
		//printf("camera1 entertc:(%f,%f)\n", 1024 * camera1Entertc.x, 1024 * camera1Entertc.y);
		float2 camera1EXittc = getCameraTc(exitReservedPos, d_modelViewRight, d_proj);
		//printf("camera1 exittc:(%f,%f)\n", 1024 * camera1EXittc.x, 1024 * camera1EXittc.y);

		float4 temp = MutiMatrixN(modelView, make_float4(enterReservedPos, 1));
		float enterZ = -temp.z;
		temp = MutiMatrixN(modelView, make_float4(exitReservedPos, 1));
		float exitZ = -temp.z;
		float step = (enter_projRatio > exit_projRatiok) ? -1.0 : 1.0;
		float enterP = min(1, max(0, enter_projRatio));
		float exitP = min(1, max(0, exit_projRatiok));
		my_printf("enter_projRatio:%f,exit_projRatiok:%f\n", enter_projRatio, exit_projRatiok);

		//printf("enterP,enterP(%f,%f)\n", enterP, exitP);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
		 dpdx = (repo(exitZ) - repo(enterZ)) / (exit_projRatiok - enter_projRatio);
	}
	float middleY = yIndex + 0.5, before = texEnd + 0.5, end = texEnd + 1 + 0.5;

	float3 beforeCenterPos = make_float3(tex2D(cudaPosTex, before, yIndex + 0.5));
	float3 beforeNormal = normalize(d_cameraPos - beforeCenterPos);
	float3 endCenterPos = make_float3(tex2D(cudaPosTex, end, yIndex + 0.5));
	float3 endNormal = normalize(d_cameraPos - endCenterPos);


	float  center_projRatiok;
	float3  centerRevervedPos, _;
	bool f_;
	rayIntersertectTriangle(posW, normalize(directionW), cameraPos, beforeCenterPos, endCenterPos, d_modelViewRight, span, &centerRevervedPos, &_, f_, center_projRatiok, _);
	float4 temp = MutiMatrixN(modelView, make_float4(centerRevervedPos.x, centerRevervedPos.y, centerRevervedPos.z, 1));
	float rayZ = -temp.z;

	float cneterP = min(1, max(0, center_projRatiok));
	float texX = texBegin + span*cneterP;
	int mappedX;
	float4 outcolor;
	int texFetchState = noMappedPosition(make_float2(texX, yIndex + 0.5), mappedX, outcolor);
	if (RAYVALID == texFetchState)
	{
		int leftX = mappedX - 1;
		int rightX = mappedX + 1;
		const int eocWidth = d_imageWidth* ROWLARGER;
		if (mfracf(texX)>0.5&&pixelValid(rightX, yIndex))
		{
			float dzdx = d_edge_buffer[yIndex*eocWidth + mappedX].dzdx;
	
		}
		else if (mfracf(texX) < 0.5&&pixelValid(leftX, yIndex))
		{

		}

	}

	return 0;
}

//如果不是一个ID,返回OHTEROBJECT，如果有交点返回INTERSECT，没有交点返回MISSINGNOTE，returnToMainC记录是是否有非冗余值
__device__ int intersectCameraID(float3 posW, float3 directionW, float3 cameraPos, ListNote currentNote, int yIndex, bool isRayUp, int occludedObjId, 
	float* modelView,// 查询modelView 相机下的深度
	bool& returnToMainC, EOCPixel &lastPixel, float4& intersectColor, float2& exitTC, float3& exitWorldPos)
{
	int texEnd = currentNote.endIndex;  // 这个是右边边界-的值
	int texBegin = currentNote.beginIndex;
	int span = texEnd + 1 - texBegin;
	int currentObjectId = nearestInt(tex2D(cudaNormalTex, texEnd - span / 2.0, yIndex + 0.5).w);
	//printf("current obj fetch (%f,%f)\n", texEnd + span / 2.0, yIndex + 0.5);
	//printf("note mid obj Id:%d\n", currentObjectId);
	//printf("texBegin,span(%d,%d)\n", texBegin, span);
	if (currentObjectId != occludedObjId)
	{
		return OHTEROBJECT;
	}
	
#define GAP 0.01
	float exitY, enterY, before = texEnd + 0.5, end = texEnd + 1 + 0.5;
	
	if (isRayUp)
	{
		exitY = yIndex + 1.0 - GAP;
		enterY = yIndex + GAP;
	}
	else
	{
		enterY = yIndex + 1.0 - GAP;
		exitY = yIndex + GAP;
	}
	
#undef GAP
	float2 beforeEnterTc = make_float2(before, enterY);                 //left
	float2 endEnterTc = make_float2(end, enterY);
	float2 beforeExitTc = make_float2(before, exitY);                 //left
	float2 endExitTc = make_float2(end, exitY);

	float3 beforeCenterPos = make_float3(tex2D(cudaPosTex, before, yIndex + 0.5));
	float3 beforeNormal = normalize(d_cameraPos-beforeCenterPos);
	float3 endCenterPos = make_float3(tex2D(cudaPosTex, end, yIndex + 0.5));
	float3 endNormal = normalize(d_cameraPos - endCenterPos);

	float3 beforeEnterPos = plateInterpolation(beforeCenterPos, beforeNormal, beforeEnterTc, d_cameraPos);
	float3 endEnterPos = plateInterpolation(endCenterPos, endNormal, endEnterTc, d_cameraPos);
	float3 beforeExitPos = plateInterpolation(beforeCenterPos, beforeNormal, beforeExitTc, d_cameraPos);
	float3 endExitPos = plateInterpolation(endCenterPos, beforeNormal, endExitTc, d_cameraPos);
	

	float enter_projRatio, exit_projRatiok;
	float3 enterReservedPos, exitReservedPos, _;
	bool f_;
	rayIntersertectTriangle(posW, normalize(directionW), cameraPos, beforeEnterPos, endEnterPos, d_modelViewRight, span, &enterReservedPos, &_, f_, enter_projRatio, _);
	rayIntersertectTriangle(posW, normalize(directionW), cameraPos, beforeExitPos, endExitPos, d_modelViewRight, span, &exitReservedPos, &_, f_, exit_projRatiok, _);
	
	float2 camera1Entertc = getCameraTc(enterReservedPos, d_modelViewRight, d_proj);
	//printf("camera1 entertc:(%f,%f)\n", 1024 * camera1Entertc.x, 1024 * camera1Entertc.y);
	float2 camera1EXittc = getCameraTc(exitReservedPos, d_modelViewRight, d_proj);
	//printf("camera1 exittc:(%f,%f)\n", 1024 * camera1EXittc.x, 1024 * camera1EXittc.y);

	float4 temp = MutiMatrixN(modelView, make_float4(enterReservedPos, 1));
	float enterZ = -temp.z;
	temp = MutiMatrixN(modelView, make_float4(exitReservedPos, 1));
	float exitZ = -temp.z;
	float step = (enter_projRatio > exit_projRatiok) ? -1.0 : 1.0;
	float enterP = min(1, max(0, enter_projRatio));
	float exitP = min(1, max(0, exit_projRatiok));
	my_printf("enter_projRatio:%f,exit_projRatiok:%f\n", enter_projRatio, exit_projRatiok);

	//printf("enterP,enterP(%f,%f)\n", enterP, exitP);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
	float dpdx = (repo(exitZ) - repo(enterZ)) / (exit_projRatiok - enter_projRatio);
	float4 color;
	
	bool isInloop =false;
	my_printf("enterP:%f,exitP:%f\n", enterP, exitP);
	EOCPixel currentPixel;
	for (float tex = (texBegin + span* enterP); tex <(texBegin + span* exitP); tex += step)
	{
		//printf("ratioOneD:%f,test tx:%f,currentZ:%f\n", ratioOneD, tex, currentZ);
		isInloop = true;
		//currentPixel = occludedRayState(tex, yIndex, texBegin, span, enterZ, dpdx, enter_projRatio,  color);
		currentPixel = occludedRayLinarState(tex, yIndex, texBegin, span, enterZ, dpdx, enter_projRatio, color);
		my_printf("current state:%d\n", currentPixel.m_state);
		my_printf("lastDepth,currentDepth:%f,%f\n", lastPixel.m_detpth, currentPixel.m_detpth);
		bool isIntersect = (RAYISUNDER == currentPixel.m_state &&lastPixel.m_state == RAYISUP) || (RAYISUP == currentPixel.m_state &&lastPixel.m_state == RAYISUNDER);
		my_printf("valid current(%d,last:%d,intersect:%d)\n", currentPixel.m_isValid, lastPixel.m_isValid, isIntersect);
		if ( currentPixel.m_isValid&& lastPixel.m_isValid  && isIntersect)
		{

			my_printf("intersect\n");
			intersectColor = color;
			//printf("color:%f,%f,%f,%f\n", intersectColor.x, intersectColor.y, intersectColor.z, intersectColor.w);
			return INTERSECT;
		}
		else 
		{
			lastPixel = currentPixel;
		}
	}
	if (!isInloop)
		lastPixel.m_isValid = false;
	/*
	exitTC.x = (texBegin + span* exitP) / d_imageWidth;
	exitWorldPos = exitReservedPos;
	if (isInLoop)
	{
	//printf("last x:%f\n", lastSearch);
	exitTC.x = lastSearch / d_imageWidth;
	float zRatio = (currentZ - enterZ) / (exitZ - enterZ);
	exitWorldPos = lerp(enterReservedPos, exitReservedPos, zRatio);
	//printf("zRatio:%f\n", zRatio);
	float2 outTc = getCameraTc(exitWorldPos, d_modelView, d_proj);
	//printf("outTc :(%f,%f)\n", 1024 * outTc.x, 1024 * outTc.y);
	}
	float2 outtc = getCameraTc(exitWorldPos, d_modelView, d_proj);
	*/
	//printf(" missingNote\n");
	if (isInloop)
		return MISSINGNOTE;
	else
		return OUTOCCLUDED;


#undef GAP

}
// 检查在objectId所代表的遮挡物下面的求交结果要么返回NOOBJECTNOTE，没有找到ID一样的要么返回misiing 没焦点
__device__ int isIntersectLineWithObjectId(float3 posW, float3 directionW, float3 cameraPos, int lineNum, bool isRayUp, int occudedObjId,
	float* modelView, bool& returnTomain, EOCPixel &previousPixel, float3& outPos, float4& resultColor)
{
	ListNote currentNote = *((ListNote*)&d_cudaPboBuffer[lineNum]);

	//printf("Render:next %d end:%d begin:%d,\n", currentNote.nextPt, currentNote.endIndex, currentNote.beginIndex);
	my_printf("testLine num:%d\n", lineNum);
	int leftEdgeIndex = 0;
	bool hasObjectNote = false;
	/*if (lineNum < 698)
	{
		return NOOBJECTNOTE;
	}*/
	while (currentNote.nextPt != 0)
	{
		int noteIndex = currentNote.nextPt;
		currentNote = d_listBuffer[currentNote.nextPt];


		//printf("intersect with a line\n");
		float2 _;
		//如果不是一个ID,返回OHTEROBJECT，如果有交点返回INTERSECT，没有交点返回MISSINGNOTE，returnToMainC记录是是否有非冗余值
		int result = intersectCameraID(posW, directionW, cameraPos, currentNote, lineNum, isRayUp, occudedObjId, modelView, returnTomain, previousPixel, resultColor, _, outPos);

		if (result != OHTEROBJECT)
		{
			//如果找到了相同obj
			//printf("obj found\n");
			hasObjectNote = true;
		}
		if (INTERSECT == result)
		{
			return INTERSECT;
		}
		else if (currentNote.nextPt == 0)
		{
			//最后一个节点
			break;
		}
		else if (OHTEROBJECT == result || MISSINGNOTE == result)
		{
			//test other note in the same line
			currentNote = *((ListNote*)&d_cudaPboBuffer[currentNote.nextPt]);
		}
		else if (OUTOCCLUDED == result)
		{
			break;
		}
	}

	if (hasObjectNote)
	{
		//返回没有收到

		my_printf("return MISSINGNOTE\n");
		return MISSINGNOTE;
	}
	my_printf("return NOOBJECTNOTE\n");
	return NOOBJECTNOTE;
}
//#define RAYISUP 0
//#define RAYOUT 1
//#define RAYISUNDER 2
//坐标都在0-1空间
__device__ int rayBelowMainTex(float n, int stepN, float2 projStart, float2 interval, float rayStartz, float rayEndZ, float2 &tc)
{
	float alpha = n / stepN;
	tc = projStart + interval* n / stepN;
	float currRayPointZ = 1 / ((1 - alpha)*(1 / rayStartz) + (alpha)*(1 / rayEndZ));
	float currSamplePointZ = colorTextreNorTc(tc).w;
	my_printf("tc:(%f,%f),n:%f,stepN:%d,z:(%f,%f)\n", 1024 * tc.x, 1024 * tc.y, n, stepN, currRayPointZ, currSamplePointZ);
	if (tc.x>(1 - 1.0 / d_imageWidth) || tc.x<(1.0 / d_imageWidth) || tc.y<(1.0 / d_imageHeight) || tc.y>(1.0 - 1.0 / d_imageHeight) || currSamplePointZ > 0)
	{
		return 0;
	}
	// 因为是-值，光线的比图片深度远（大）意味着z要小
	else if (currRayPointZ <= currSamplePointZ)
		return RAYISUNDER;
	else
		return RAYISUP;
}

__device__ float3 startPointOfTex(float2 projStart, float2 projEnd, float rayStartz, float rayEndz)
{
	float2 wtx = projEnd - projStart;
	float alpha = 0;
	if (projStart.x<0)
	{
		alpha = (0.00000 - projStart.x) / (projEnd.x - projStart.x);
		projStart.y = projStart.y + (0.00000 - projStart.x)*wtx.y / wtx.x;
		projStart.x = 0.00000;
	}
	else if (projStart.x>1)
	{
		alpha = (1.0f - projStart.x) / (projEnd.x - projStart.x);
		projStart.y = projStart.y + (1.0f - projStart.x)*wtx.y / wtx.x;
		projStart.x = 1.0f;

	}
	else if (projStart.y>1)
	{
		alpha = (1.0f - projStart.y) / (projEnd.y - projStart.y);

		projStart.x = projStart.x + (1.0f - projStart.y)*wtx.x / wtx.y;
		projStart.y = 1.0f;

	}
	else if (projStart.y<0)
	{
		alpha = (0.0f - projStart.y) / (projEnd.y - projStart.y);

		projStart.x = projStart.x + (0.00000 - projStart.y)*wtx.x / wtx.y;
		projStart.y = 0.00000;

	}
	rayStartz = 1 / ((1 - alpha)*(1 / rayStartz) + (alpha)*(1 / rayEndz));
	alpha = 0;
	if (projStart.x<0)
	{
		alpha = (0.00000 - projStart.x) / (projEnd.x - projStart.x);

		projStart.y = projStart.y + (0.00000 - projStart.x)*wtx.y / wtx.x;
		projStart.x = 0.00000;

	}
	else if (projStart.x>1)
	{
		alpha = (1.0f - projStart.x) / (projEnd.x - projStart.x);
		projStart.y = projStart.y + (1.0f - projStart.x)*wtx.y / wtx.x;
		projStart.x = 1.0f;

	}
	else if (projStart.y>1)
	{
		alpha = (1.0f - projStart.y) / (projEnd.y - projStart.y);

		projStart.x = projStart.x + (1.0f - projStart.y)*wtx.x / wtx.y;
		projStart.y = 1.0f;

	}
	else if (projStart.y<0)
	{
		alpha = (0.0f - projStart.y) / (projEnd.y - projStart.y);

		projStart.x = projStart.x + (0.00000 - projStart.y)*wtx.x / wtx.y;
		projStart.y = 0.00000;

	}
	rayStartz = 1 / ((1 - alpha)*(1 / rayStartz) + (alpha)*(1 / rayEndz));

	return make_float3(projStart, rayStartz);

}
__device__ int intersetctNoteGraph()
{

}
__device__ int intersectTexRay(float3 posW, float3 directionW, float beginOffset,float endOffset, float4& oc)
{
	float2 d_mapScale = 1.0 / make_float2(d_construct_width, d_construct_height);
	float3 rayStart, rayEnd;
	float4 color;
	//printf("posW:(%f,%f,%f,1)\n", posW.x, posW.y, posW.z);
	float4 posWE = MutiMatrixN(d_modelView, make_float4(posW, 1));
	float4 temp2 = MutiMatrixN(d_proj, posWE);
	temp2 = temp2 / temp2.w;
	//printf("temp2:%f,%f", (temp2.x*0.5 + 0.5) * 1024, (temp2.y*0.5 + 0.5) * 1024);
	float3 posW3 = toFloat3(posWE);
	float4 temp = MutiMatrixN(d_modelView, make_float4(directionW, 0));
	float3 RE = normalize(make_float3(temp.x, temp.y, temp.z));
	//printf("RE:(%f,%f,%f,1)\n", RE.x, RE.y, RE.z);
	float epison = beginOffset;
	rayStart = posW3 + RE*epison;

	float max_rfl = endOffset;//far*diffuseColor.w;
	rayEnd = posW3 + RE*max_rfl;

	//p.color0.xy = tc;
	if (rayEnd.z>0)
	{
		float step = -posW3.z / RE.z;
		rayEnd = posW3 + RE*(step - 1);
	}
	temp = MutiMatrixN(d_proj, make_float4(rayStart, 1));

	float3 projStart = toFloat3(temp);
	temp = MutiMatrixN(d_proj, make_float4(rayEnd, 1));
	float3 projEnd = toFloat3(temp);

	projStart.x = 0.5*projStart.x + 0.5;
	projEnd.x = 0.5*projEnd.x + 0.5;
	projStart.y = 0.5*projStart.y + 0.5;
	projEnd.y = 0.5*projEnd.y + 0.5;

	float3 shiftValue = startPointOfTex(make_float2(projStart.x, projStart.y), make_float2(projEnd.x, projEnd.y), rayStart.z, rayEnd.z);
	projStart.x = shiftValue.x;
	projStart.y = shiftValue.y;
	rayStart.z = shiftValue.z;

	//printf("projStart(%f,%f),projEnd(%f,%f)\n", (projStart.x) * 1024, (projStart.y) * 1024, projEnd.x * 1024, projEnd.y * 1024);


	//oc = make_float4(projStart.x,projStart.y,projStart.z,0.7);	
	//return 1;
	float2 interval = make_float2(projEnd.x, projEnd.y) - make_float2(projStart.x, projStart.y);
	int stepN;
	//printf("interval:(%f,%f)\n", interval.x, interval.y);
	//printf("1024*d_mapScale:(%f,%f)\n", d_mapScale.x*1024,d_mapScale.y*1024);
	if (abs(interval.x)>abs(interval.y))
		stepN = abs(interval.x) / d_mapScale.x + 1;
	else
		stepN = abs(interval.y) / d_mapScale.y + 1;

	float currSamplePointZ, currRayPointZ, prevSamplePointZ, prevRayPointZ;
	float3 currSamplePoint, currRayPoint;
	float n = 0;
	float2 tc;
	bool isNotValid = true;


	//printf("interval*1024(%f,%f)\n", (interval.x) * 1024, (interval.y) * 1024);
	//printf("stepN:%d\n", stepN);
	int prevState, currentRayState = rayBelowMainTex(n, stepN, make_float2(projStart.x, projStart.y), interval, rayStart.z, rayEnd.z, tc);
	if (RAYOUT == currentRayState)
	{
		return false;// 没在相机空间内
	}
	if (stepN<2)
	{
		oc = colorTextreNorTc(make_float2(projStart.x, projStart.y) + interval / 2);
		return 1;
	}
	for (; n <= stepN; n += 1)
	{
		prevState = currentRayState;
		currentRayState = rayBelowMainTex(n, stepN, make_float2(projStart.x, projStart.y), interval, rayStart.z, rayEnd.z, tc);
		if (RAYOUT == currentRayState)
		{
			my_printf("RAYOUT\n");
			return false;// 没在相机空间内
		}
		if (RAYISUNDER == currentRayState && RAYISUP == prevState)
		{
			
			color = colorTextreNorTc(tc);
			float lastAlpha = 0;
			if (n >= 1)
				lastAlpha = (float)(n - 1) / stepN;
			float2 lastTc = make_float2(projStart.x, projStart.y) + interval* lastAlpha;
			int previewsN = n;
			my_printf("here\n");
			if (isRightEdge(lastTc) && isOccluedeArea(tc))// 如果倒数第二个是在边横沿上，也就是进入遮挡体
			{
			
				int stepY = abs(interval.y) / d_mapScale.y + 1;
				float3 exitPos;
				bool rayAdvanced = false;
				
					//找到遮挡体的bojectId
				float2 nonNorTc = tc* make_float2(d_imageWidth, d_imageHeight);
				int objectId = nearestInt(tex2D(cudaNormalTex, nonNorTc.x, nonNorTc.y).w);
				bool isUp = interval.y > 0;
				int startLineNum = floor(lastTc.y*d_imageHeight);  // 如果是6.6 行，按第6行搜索 单纯取整
				float3 outPos;
				bool returntoMain = false;
				// 检查在objectId所代表的遮挡物下面的求交结果要么返回NOOBJECTNOTE，没有找到ID一样的要么返回misiing 没焦点
				EOCPixel lastPicel(RAYISUP, 0.0f);
				lastPicel.m_isValid = true;
				//EOCPixel lastPicel(RAYBEGIN, 0.0f);
				int result = isIntersectLineWithObjectId(posW, directionW, d_eocPos, startLineNum, isUp, objectId, d_modelViewRight, returntoMain, lastPicel, outPos, oc);
				int lineId = startLineNum + (isUp ? 1 : -1);// lineId 存储的是要搜索的下一行
				n = previewsN;
				int outLineId = startLineNum;
				if (false == returntoMain)
				{
					outLineId = startLineNum + (isUp ? 1 : -1);
					n += (float)stepN / stepY;
				}
				my_printf("out line id increase: %d\n", outLineId);
				while (result == MISSINGNOTE)// 如果这一列中没有求交到MISS,并且没有发生
				{

					result = isIntersectLineWithObjectId(posW, directionW, d_eocPos, lineId, isUp, objectId, d_modelViewRight, returntoMain, lastPicel, outPos, oc);
					lineId = lineId + (isUp ? 1 : -1);
					n += (float)stepN / stepY;
						
					my_printf("out line id increase: %d\n", outLineId);
					my_printf("result: %d\n", result);

				}
				if (result == INTERSECT)
				{
					my_printf("addition note intersect\n");
					my_printf("oc:(%f,%f,%f)\n",oc.x,oc.y,oc.z);
					return 1;
				}
				else //如果到了一重新回到主相机
				{
					//通过lineId 的号码来判断n更新n的地方，为该行的一个偏移处，
					float newN;
#define GAP 0.01
					newN = stepN * abs(outLineId + GAP - projStart.y / d_mapScale.y) / (abs(interval.y) / d_mapScale.y);
					my_printf("new outlineId:%d\n", outLineId);
					n = max(newN, previewsN);
					currentRayState = rayBelowMainTex(n, stepN, make_float2(projStart.x, projStart.y), interval, rayStart.z, rayEnd.z, tc);
#undef GAP
					continue;// 下一步寻找
				}
				
			}
			else if (isTopEdge(lastTc) )
			{
				my_printf("top intersection\n");
				oc = make_float4(1,0,1,0);
				return 1;
			}
			// 如果倒数第二个是边界而本身不是遮挡体说明从遮挡体出来的光线
			color.w = 1;
			oc = color;
			my_printf("main camera intersection found\n");
			return 1;
		}
	}
	// 在搜索范围内找不到
	return 0;

}

__global__ void construct_kernel(int kernelWidth, int kernelHeight)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= kernelWidth || y >= kernelHeight)
		return;
#ifdef PRINTDEBUG
	if (x != 657 || y != 430)
	   return;
#endif
	//if ( y <100)
	//	return;
	//if ( y >= 470||x>=414)
	//	return;
	//printf("test:x%d,y:%d\n", x, y);
	const int index = y*kernelWidth + x;
	float2 tc = make_float2(x + 0.5, y + 0.5) / make_float2(kernelWidth, kernelHeight);
	float3 beginPoint = getImagePos(tc, d_modeView_inv_construct);
	my_printf("beginPoint:(%f,%f,%f)\n", beginPoint.x, beginPoint.y, beginPoint.z);
	float3 viewDirection = normalize(beginPoint - d_construct_cam_pos);

	my_printf("viewDirection:(%f,%f,%f)\n", viewDirection.x, viewDirection.y, viewDirection.z);
	float4 outColor;
	if (intersectTexRay(beginPoint, viewDirection,BEGINOFFSET,ENDOFFSET, outColor))
	{
		//printf("here outcolor(%f,%f,%f,%f)\n", outColor.x, outColor.y, outColor.z, outColor.w);
		d_cuda_construct_texture[index] = make_float4(outColor.x, outColor.y, outColor.z, 1);//tex2D(cudaColorTex, x, y);
	}
	else
	{
		d_cuda_construct_texture[index] = make_float4(0, 0, 0, 1);//tex2D(cudaColorTex, x, y);
	}
}
void construct_cudaInit()
{
	checkCudaErrors(cudaMalloc(&modelView_construct, 16 * sizeof(float)));
	checkCudaErrors(cudaMalloc(&project_construct, 16 * sizeof(float)));
	checkCudaErrors(cudaMalloc(&modelView_construct_inv, 16 * sizeof(float)));


}
__device__ bool eocIsRight(int x,int y)
{
	float4 value = tex2D(optixColorTex, x, y);
	return value.w > 0;
}
#define EDGETHRES 0.005
#define ISMIDDLE 3
#define ISRIGHT 4
#define ISLEFT 5
__device__ edgeInfo isEocRightEdge(int left, int ceter, int right, int y, int type)
{
	edgeInfo value;
	float leftRe = 1.0 / tex2D(optixColorTex, left, y).w;
	float centerRe = 1.0 / tex2D(optixColorTex, ceter, y).w;
	float rightRe = 1.0 / tex2D(optixColorTex, right, y).w;
	my_printf("left:%f,center:%f,right:%f,value:%f\n", leftRe, centerRe, rightRe, abs(2 * centerRe - leftRe - rightRe));

	if (ISMIDDLE == type)
	{
		value.dzdx = (rightRe - leftRe) / 2;
	}
	else if (ISLEFT == type)
	{
		value.dzdx = centerRe - leftRe;
	}
	else if (ISRIGHT == type)
	{
		value.dzdx = rightRe-centerRe;
	}
	value.isRightEdge = abs(2 * centerRe - leftRe - rightRe) > EDGETHRES;
	return value;
}
__global__ void EocEdgeKernel(int kernelWidth, int kernelHeight)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if (x >= kernelWidth || y >= kernelHeight)
		return;
#ifdef PRINTDEBUG
	if (x != 649 || y != 774)
		return;
#endif
	if (eocIsRight(x, y) && eocIsRight(x - 1, y) && eocIsRight(x + 1, y))
	{
		my_printf("center edge\n");
		d_edge_buffer[y*kernelWidth + x] = isEocRightEdge(x - 1, x, x + 1, y,ISMIDDLE);
	}
	else if  (eocIsRight(x, y) && eocIsRight(x +1 , y) && eocIsRight(x + 2, y))
	{
		d_edge_buffer[y*kernelWidth + x] = isEocRightEdge(x, x + 1, x + 2, y,ISLEFT);
	}
	else if (eocIsRight(x, y) && eocIsRight(x - 1, y) && eocIsRight(x -2, y))
	{
		d_edge_buffer[y*kernelWidth + x] = isEocRightEdge(x - 2, x - 1, x, y,ISRIGHT);
	}
	else
	{
		d_edge_buffer[y*kernelWidth + x].isRightEdge = false;
	}
	
}

void edgeRendering(int width, int height)
{
	dim3 blockSize(16, 16, 1);
	dim3 gridSize(width / blockSize.x, height / blockSize.y, 1);
	EocEdgeKernel << <gridSize, blockSize >> >(width, height);
}
void cuda_Construct(int width, int height)
{
	dim3 blockSize(16, 16, 1);
	dim3 gridSize(width / blockSize.x, height / blockSize.y, 1);
	construct_kernel << <gridSize, blockSize >> >(width, height);
}
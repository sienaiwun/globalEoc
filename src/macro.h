#define SCREEN_WIDTH 1024
#define SCREEN_HEIGHT 1024
#define K 8
#define OPTIX
//�����ұ������eoc�������������
#define  DIS_ORIGIN_W  5  
//�����ϱ������eoc�������������
#define DIS_ORIGIN_H 5
#define FAR_PLANE_DIS (500.0)
#define NEAR_PLANE_DIS (1)
//����
#define  to_flocus  25
#ifdef OPTIX
	#define RAYTRACINGPATH "./output/raytracing.ptx"
	#define DIFFUSEPATH "./output/diffuse.ptx"
	#define TEXTUREPATH "./output/textureobj.ptx"
	#define TRIAGNELPATH  "./output/triangleMesh.ptx"
	#define CONSTRUCTPATH   "./output/constructKernel.ptx"
#endif

#ifndef MACRO_H
#define MACRO_H
typedef enum  Ray_type
{
	primary_ray_e,
	reflected_ray_e,
} Ray_type;
#endif

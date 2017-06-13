#define SCREEN_WIDTH 1024
#define SCREEN_HEIGHT 1024
#define K 8
#define OPTIX
//定义右边相机到eoc相机主相机距离宏
#define  DIS_ORIGIN_W  5  
//定义上边相机到eoc相机主相机距离宏
#define DIS_ORIGIN_H 5
#define FAR_PLANE_DIS (500.0)
#define NEAR_PLANE_DIS (1)
//定义
#define  to_flocus  25
#ifdef OPTIX
	#define RAYTRACINGPATH "./output/raytracing.ptx"
	#define DIFFUSEPATH "./output/diffuse.ptx"
	#define TEXTUREPATH "./output/textureobj.ptx"
	#define TRIAGNELPATH  "./output/triangleMesh.ptx"
#endif

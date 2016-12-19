#define SCREEN_WIDTH 1024
#define SCREEN_HEIGHT 1024
#define K 8
#define OPTIX
#define  dis_orgin  5
#define  to_flocus  25
#ifdef OPTIX
	#define RAYTRACINGPATH "./output/raytracing.ptx"
	#define DIFFUSEPATH "./output/diffuse.ptx"
	#define TEXTUREPATH "./output/textureobj.ptx"
	#define TRIAGNELPATH  "./output/triangleMesh.ptx"
#endif

#pragma once
#include "textureManager.h"
#include "glm.h"
#include <map>
#include <vector>
//#include <GL/glew.h>
#include <glm.hpp>
#include <algorithm>
#include <nvMath.h>
#include "glslShader.h"
#include "textureManager.h"
#include "macro.h"
#ifdef OPTIX
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
using namespace optix;
#include <cuda.h>
#include <cuda_runtime.h>

#endif




template <class TYPE_NAME>
class Triple
{
public:
	TYPE_NAME x, y, z;

public:
	Triple(TYPE_NAME _x, TYPE_NAME _y, TYPE_NAME _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}

	TYPE_NAME& operator[](int id)
	{
		if (id == 0) return x;
		if (id == 1) return y;
		return z;
	}


	bool operator==(const Triple &p) const
	{
		return x == p.x&&y == p.y && z == p.z;
	}
	bool operator<(const Triple &p) const
	{
		if (x != p.x) return x < p.x;
		if (y != p.y) return y < p.y;
		return z < p.z;
	}


}; 

class myGeometry{
public:
	myGeometry(const char* fileName);
	int getPositionSize() const;
	int getNormalSize() const;
	int getTexCoordSize() const;
	static void drawQuad(glslShader &shader, bool addition , nv::vec2f beginPoint = nv::vec2f(0, 0), nv::vec2f endPoint = nv::vec2f(1, 1));
	static void drawQuad(glslShader & shader);
	static void drawQuadMesh(glslShader & shader, int w, int h);
	static GLuint initImageMesh(int w, int h);
	static void  drawTriangle(nv::vec3f newOrigin,  glslShader& shader);
	void render(glslShader & shader, textureManager & manager);
	//
	//compiled data access functions
	//
	const float* getCompiledVertices() const;
	const GLuint* getCompiledGroupIndices(int gID) const;
	const GLuint* getCompiledIndices() const;

	int getCompiledPositionOffset() const;
	int getCompiledNormalOffset() const;
	int getCompiledTexCoordOffset() const;

	// returns the size of the merged vertex in # of floats
	int getCompiledVertexSize() const;

	int getCompiledVertexCount() const;
	int getCompiledGroupIndexCount(int gID) const;
	int getCompiledIndexCount() const;
	int getMaterialsCount() const{ return _materials.size(); }

	GLuint getVBO() const;
	GLuint getIBO() const;
	GLuint getGroupIBO(int gID) const;
	const GLMmaterial & getGroupMaterial(int gID) const;
	int getGroupCount() const;

	void getVertexStartEnd(int & start, int & end)
	{
		start = _baseVertexID;
		end = _baseVertexID + _vertices.size() / _vtxSize;
	}

	void getModelAABB(nv::vec3f& Mmin, nv::vec3f& Mmax)
	{
		Mmin.x = std::min<float> (Mmin.x, _modelAABBmin.x);
		Mmin.y = std::min<float>(Mmin.y, _modelAABBmin.y);
		Mmin.z = std::min<float>(Mmin.z, _modelAABBmin.z);

		Mmax.x = std::max<float>(Mmax.x, _modelAABBmax.x);
		Mmax.y = std::max<float>(Mmax.y, _modelAABBmax.y);
		Mmax.z = std::max<float>(Mmax.z, _modelAABBmax.z);
	}

	inline const glm::vec3 & getModelCenter(){ return _modelCenter; };

	int Create_GPU_Buffer();

	int CreateDynamicGPU_Buffer();

	void adjustIndicesAndGroupMaterial(int baseVerticesID, int baseGroupMaterialID);

	void appendDynamicObject(myGeometry& dynamicObject);

	inline int getSize() 
	{
		return _groupIndices.size();
	}
	static void setTexManager(textureManager* p);
#ifdef OPTIX
	optix::GeometryInstance getInstance(int index);//myGeometry has many group
	optix::Geometry getOptixGeometry(int index);
	optix::Material getMeterial(int index);
	
	static optix::Context* p_rtContext;
	static optix::TextureSampler  getTexture(const char * fileName);
#endif
protected:
	//Would all this be better done as a channel abstraction to handle more arbitrary data?


	//data structures optimized for rendering, compiled model
	std::vector<std::vector<GLuint> > _groupIndices;
	std::vector<GLuint> _indices;
	
	
	std::vector<float> _vertices;
	std::vector<GLMmaterial> _materials;

	std::vector<GLuint> _groupIBO;
	std::vector<int> _groupMaterial;
	GLuint _VBO;
	GLuint _IBO;

	int _pOffset;
	int _nOffset;
	int _tcOffset;
	int _vtxSize;

	int _posSize;
	int _normalSize;
	int _tcSize;

	int _groupNum;

	bool _mainModel;

	int _baseVertexID;

	nv::vec3f _modelAABBmin, _modelAABBmax;
	glm::vec3 _modelCenter;


private:
	GLMmodel * _glmodel;
};
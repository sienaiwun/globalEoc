#include "myGeometry.h"
#include <map> 
#include <crtdbg.h>
#include <windows.h>
#include <olectl.h>              
#include <math.h>  
#ifdef OPTIX
optix::Context* myGeometry::p_rtContext = NULL;
#endif
textureManager* pTextureManager = NULL;
myGeometry::myGeometry(const char* fileName/*const GLMmodel * glmodel*/) :_mainModel(true), _baseVertexID(0)
{
	_glmodel = glmReadOBJ(fileName);
	
	


	if (!_glmodel)
	{
		printf("cpu model [%s] is bad!\n", fileName);
		exit(-1);
	}
	/*glmFacetNormals(_glmodel);
	glmVertexNormals(_glmodel, 50.0f);*/

	

	float *positions = _glmodel->vertices;
	float *normals = _glmodel->normals;
	float *texCoords = _glmodel->texcoords;

	_posSize = 3;
	_normalSize = 3;
	_tcSize = 2;
	_pOffset = 0;
	_nOffset = _pOffset + _posSize;
	_tcOffset = _nOffset + _normalSize;
	_vtxSize = _posSize + _normalSize + _tcSize;
	
	float temp[4] = { 0, 0, 0, 0 };

	_groupIndices.clear();
	GLMgroup * group = _glmodel->groups;
	GLMtriangle * triangle = _glmodel->triangles;
	std::map<Triple<GLuint>, int> vertexIdx;
	int cnt = 0;
	_indices.clear();
	int groupId = 0;
	_groupNum = 0;
	while (group){
		if( group->numtriangles > 0)
			_groupNum++;
		group = group->next;
	}

	group = _glmodel->groups;

	for (; group; group = group->next)
	{
		if (group->numtriangles <=0)
			continue;
		_groupIndices.resize(cnt+1);
		_groupMaterial.push_back(group->material);

		for (int i = 0; i < group->numtriangles; i++)
		{
			unsigned int tid = group->triangles[i];
			GLMtriangle curTriangle = triangle[tid];
			for (int j = 0; j < 3; j++)
			{
				Triple<GLuint> key(curTriangle.vindices[j], curTriangle.nindices[j], curTriangle.tindices[j]);
				if (!vertexIdx.count(key))
				{
					// make vertex;
					int index = curTriangle.vindices[j];
					if (positions != NULL)
						_vertices.insert(_vertices.end(), positions + index*_posSize, positions + index* _posSize + _posSize);
					else
						_vertices.insert(_vertices.end(), temp, temp + _posSize);
					index = curTriangle.nindices[j];
					if (normals != NULL)
						_vertices.insert(_vertices.end(), normals + index* _normalSize, normals + index * _normalSize + _normalSize);
					else
						_vertices.insert(_vertices.end(), temp, temp + _normalSize);
					index = curTriangle.tindices[j];
					if (texCoords != NULL)
						_vertices.insert(_vertices.end(), texCoords + index * _tcSize, texCoords + index * _tcSize + _tcSize);
					else
						_vertices.insert(_vertices.end(), temp, temp+ _tcSize);

					int value = (_vertices.size() / _vtxSize) - 1;
					vertexIdx[key] = value;
					_groupIndices[cnt].push_back(value);
					_indices.push_back(value);
				}
				else
				{
					int value = vertexIdx[key];
					_groupIndices[cnt].push_back(value);
					_indices.push_back(value);
				}
			}
		}

		cnt++;
		// show process status;
		groupId++;
		printf("\r                                               ");
		printf("\rconvert to GPU format: %.1f%%", (float)groupId / _groupNum * 100);
	}
	printf("\n");

	printf("mode have [%d] triangle, [%d] vertex\n", _indices.size() / 3, _vertices.size() / _vtxSize);


	// copy materials;
	for (int i = 0; i < _glmodel->nummaterials; i++)
		_materials.push_back(_glmodel->materials[i]);


	// compute model AABB on cpu;
	glmBoundingBox(_glmodel, &_modelAABBmin.x, &_modelAABBmax.x);
	nv::vec3f middle = (_modelAABBmax + _modelAABBmin) *0.5f;
	_modelCenter = glm::vec3(middle.x, middle.y, middle.z);


	// delte _glmodel;
	glmDelete(_glmodel);
}

const float* myGeometry::getCompiledVertices() const
{
	return &_vertices[0];
}

const GLuint* myGeometry::getCompiledGroupIndices(int gID) const
{
	return &_groupIndices[gID][0];
}

int myGeometry::getCompiledPositionOffset() const
{
	return _pOffset;
}

int myGeometry::getCompiledNormalOffset() const
{
	return _nOffset;
}

int myGeometry::getCompiledTexCoordOffset() const
{
	return _tcOffset;
}

int myGeometry::getCompiledVertexSize() const
{
	return _vtxSize;
}

int myGeometry::getCompiledVertexCount() const
{
	return _vertices.size() / _vtxSize;
}

int myGeometry::getCompiledGroupIndexCount(int gID) const
{
	return _groupIndices[gID].size();
}

const GLuint* myGeometry::getCompiledIndices() const
{
	return &_indices[0];
}

int myGeometry::getCompiledIndexCount() const
{
	return _indices.size();
}

#define CHECK_ERRORS()         \
	do {                         \
	GLenum err = glGetError(); \
	if (err) {                                                       \
	printf( "GL Error %d at line %d of FILE %s\n", (int)err, __LINE__,__FILE__);       \
	exit(-1);                                                      \
			}                                                                \
			} while(0)
void myGeometry::render(glslShader & shader,textureManager& manager)
{
	glBindBuffer(GL_ARRAY_BUFFER, getVBO());

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	CHECK_ERRORS();
	glVertexAttribPointer(0,
		getPositionSize(),
		GL_FLOAT,
		GL_FALSE,
		getCompiledVertexSize()*sizeof(float),
		(void*)(getCompiledPositionOffset()*sizeof(float)));

	glVertexAttribPointer(1,
		getNormalSize(),
		GL_FLOAT,
		GL_TRUE,
		getCompiledVertexSize()*sizeof(float),
		(void*)(getCompiledNormalOffset()*sizeof(float)));

	glVertexAttribPointer(2,
		getTexCoordSize(),
		GL_FLOAT,
		GL_FALSE,
		getCompiledVertexSize()*sizeof(float),
		(void*)(getCompiledTexCoordOffset()*sizeof(float)));

	CHECK_ERRORS();


	for (int gID = 0; gID < getGroupCount(); gID++)
	{	
		if (_materials.size()>0)
		shader.setMaterial(getGroupMaterial(gID), manager);	
		CHECK_ERRORS();
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, getGroupIBO(gID));
		glDrawElements(GL_TRIANGLES, getCompiledGroupIndexCount(gID), GL_UNSIGNED_INT, 0);
		CHECK_ERRORS();
	}

	CHECK_ERRORS();
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	CHECK_ERRORS();
	
}

void myGeometry::setTexManager(textureManager* p)
{
	pTextureManager = p;
}
#ifdef OPTIX
optix::Geometry myGeometry::getOptixGeometry(int i)
{
	optix::Context    rtContext = *p_rtContext;
	optix::Geometry geometry = rtContext->createGeometry();
	int indexNum = _groupIndices[i].size();
	
	geometry->setPrimitiveCount(indexNum / 3);
	geometry->setIntersectionProgram(rtContext->createProgramFromPTXFile(TRIAGNELPATH, "mesh_intersect"));
	geometry->setBoundingBoxProgram(rtContext->createProgramFromPTXFile(TRIAGNELPATH, "mesh_bounds"));

	int num_vertices = getCompiledVertexCount();
	optix::Buffer vertex_buffer = rtContext->createBufferFromGLBO(RT_BUFFER_INPUT, _VBO);
	vertex_buffer->setFormat(RT_FORMAT_USER);
	vertex_buffer->setElementSize(getCompiledVertexSize() * sizeof(float));
	vertex_buffer->setSize(num_vertices);
	geometry["vertex_buffer"]->setBuffer(vertex_buffer);

	optix::Buffer index_buffer = rtContext->createBufferFromGLBO(RT_BUFFER_INPUT, _groupIBO[i]);
	index_buffer->setFormat(RT_FORMAT_INT3);
	int indexCount = indexNum / 3;
	index_buffer->setSize(indexCount);
	geometry["index_buffer"]->setBuffer(index_buffer);

	optix::Buffer material_buffer = rtContext->createBuffer(RT_BUFFER_INPUT);
	material_buffer->setFormat(RT_FORMAT_UNSIGNED_INT);
	material_buffer->setSize(indexNum / 3);
	void* material_data = material_buffer->map();
	memset(material_data, 0, indexNum / 3 * sizeof(unsigned int));
	material_buffer->unmap();
	geometry["material_buffer"]->setBuffer(material_buffer);
	return geometry;
}
optix::TextureSampler  myGeometry::getTexture(const char * fileName)
{
	optix::Context    rtContext = *p_rtContext;

	std::string fileString = std::string(fileName);
	dimension dim = pTextureManager->m_nameToDimention[fileString];
	GLuint texId = pTextureManager->m_nameToTexId[fileString];


	glEnable(GL_TEXTURE_2D);
	BYTE *pTexture = NULL;
	pTexture = new BYTE[dim.getWidth()*dim.getHeight() * 3];
	memset(pTexture, 0, dim.getWidth()*dim.getHeight() * 3 * sizeof(BYTE));

	glBindTexture(GL_TEXTURE_2D, texId);//TexPosId   PboTex

	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pTexture);

	
	glBindTexture(GL_TEXTURE_2D, 0);//TexPosId   Pbo

	int lWidthPixels = dim.getWidth();
	int lHeightPixels = dim.getHeight();
	optix::Buffer floor_data = rtContext->createBuffer(RT_BUFFER_INPUT);
	floor_data->setFormat(RT_FORMAT_FLOAT4);
	floor_data->setSize(lWidthPixels, lHeightPixels);
	float4* tex_data = (float4*)floor_data->map();
	uchar3* pixel_data = (uchar3*)pTexture;
	for (unsigned int i = 0; i < lWidthPixels * lHeightPixels; ++i) {
		tex_data[i] = make_float4(static_cast<float>(pixel_data[i].x) / 255.99f,
			static_cast<float>(pixel_data[i].y) / 255.99f,
			static_cast<float>(pixel_data[i].z) / 255.99f,
			1.f);
	}
	floor_data->unmap();


	optix::TextureSampler floor_tex = rtContext->createTextureSampler();
	floor_tex->setWrapMode(0, RT_WRAP_REPEAT);
	floor_tex->setWrapMode(1, RT_WRAP_REPEAT);
	floor_tex->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	floor_tex->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
	floor_tex->setMaxAnisotropy(1.0f);
	floor_tex->setMipLevelCount(1u);
	floor_tex->setArraySize(1u);
	floor_tex->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	floor_tex->setBuffer(0, 0, floor_data);
	
	if (pTexture)
		delete[] pTexture;
	return floor_tex;




}
optix::Material myGeometry::getMeterial(int index)
{
	assert(pTextureManager);
	const GLMmaterial & material = getGroupMaterial(index);
	const char * diffuseName = material.diffuse_map;
	int  texid = pTextureManager->getTexId(diffuseName);
	optix::Context    rtContext = *p_rtContext;
	CHECK_ERRORS();
	//	printf("diffuse map:%s\n", material.diffuse_map);
	if (texid > 0)
	{
		optix::Material flat_tex = rtContext->createMaterial();
		flat_tex->setClosestHitProgram(0, rtContext->createProgramFromPTXFile(TEXTUREPATH, "closest_hit_radiance"));
		flat_tex["diffuse_texture"]->setTextureSampler(getTexture(diffuseName));
		return flat_tex;
	}
	{
		optix::Material diffuse = rtContext->createMaterial();
		diffuse->setClosestHitProgram(0, rtContext->createProgramFromPTXFile(DIFFUSEPATH, "closest_hit_radiance"));
		diffuse["diffuse_Color"]->setFloat(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
		return diffuse;
	}

}
optix::GeometryInstance  myGeometry::getInstance(int index)
{

	
	optix::Context    rtContext = *p_rtContext;
	Material  m = getMeterial(index);
	optix::Geometry sofaGemetry = getOptixGeometry(index);
	optix::GeometryInstance instance = rtContext->createGeometryInstance();
	instance->setMaterialCount(1);
	instance->setMaterial(0, m);
	instance->setGeometry(sofaGemetry);
	return instance;
}
#endif
int myGeometry::Create_GPU_Buffer()
{
	if (!_mainModel)
	{
		printf("can not create all gpu buffer for model except main model\n");
		return 1;
	}
	printf("transfer to gpu memory...\n");
	_groupIBO.resize(_groupIndices.size());

	glGenBuffers(1, &_VBO);
	glBindBuffer(GL_ARRAY_BUFFER, _VBO);
	glBufferData(GL_ARRAY_BUFFER,
		_vertices.size()*sizeof(float),
		&_vertices[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &_IBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _IBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
		_indices.size()*sizeof(GLuint),
		&_indices[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	for (int i = 0; i < _groupIndices.size(); i++)
	{
		glGenBuffers(1, &_groupIBO[i]);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _groupIBO[i]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER,
			_groupIndices[i].size()*sizeof(GLuint),
			&_groupIndices[i][0], GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}
	return 0;
}

int myGeometry::CreateDynamicGPU_Buffer()
{
	glGenBuffers(1, &_VBO);
	glBindBuffer(GL_ARRAY_BUFFER, _VBO);
	glBufferData(GL_ARRAY_BUFFER,
		_vertices.size()*sizeof(float),
		&_vertices[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	return 0;
}

GLuint myGeometry::getVBO() const
{
	return _VBO;
}

GLuint myGeometry::getIBO() const
{
	return _IBO;

}

GLuint myGeometry::getGroupIBO(int gID) const
{
	return _groupIBO[gID];
}

int myGeometry::getPositionSize() const
{
	return 3;
}

int myGeometry::getNormalSize() const
{
	return 3;
}

int myGeometry::getTexCoordSize() const
{
	return 2;
}

const GLMmaterial& myGeometry::getGroupMaterial(int gID) const
{
	int id = _groupMaterial[gID];
	const GLMmaterial& meterial = (_materials[id]);
	//intf("%s\n", meterial.diffuse_map);
	return meterial;
}

int myGeometry::getGroupCount() const
{
	return _groupIndices.size();
}

void myGeometry::adjustIndicesAndGroupMaterial(int baseVerticesID, int baseGroupMaterialID)
{
	// this means not main model;
	_mainModel = false;
	_baseVertexID = baseVerticesID;

	for (int gid = 0; gid < _groupIndices.size(); gid++)
		for (int i = 0; i < _groupIndices[gid].size(); i++)
			_groupIndices[gid][i] += baseVerticesID;

	for (int i = 0; i < _indices.size(); i++)
		_indices[i] += baseVerticesID;

	for (int i = 0; i < _groupMaterial.size(); i++)
		_groupMaterial[i] += baseGroupMaterialID;

}
void myGeometry::drawQuad(glslShader& shader, bool addition, nv::vec2f beginPoint, nv::vec2f endPoint)
{
	if (addition)
		glDisable(GL_DEPTH_TEST);
	beginPoint = beginPoint * 2 - nv::vec2f(1, 1);
	endPoint = endPoint * 2 - nv::vec2f(1, 1);
	shader.begin();
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glVertex2fv(beginPoint);
	glVertex2f(endPoint.x, beginPoint.y);
	glVertex2fv(endPoint);
	glVertex2f(beginPoint.x, endPoint.y);
	glEnd();
	shader.end();
	if (addition)
		glEnable(GL_DEPTH_TEST);
}
void myGeometry::drawQuad(glslShader& shader)
{
	shader.begin();
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glVertex2f(-1.0f, -1.0f);
	glVertex2f(1.0f, -1.0f);
	glVertex2f(1.0f, 1.0f);
	glVertex2f(-1.0f, 1.0f);
	glEnd();
	shader.end();
}
std::map<nv::vec2i, GLuint> g_drawIndexMap;
GLuint myGeometry::initImageMesh(int w, int h)
{
	nv::vec2i index = nv::vec2i(w, h);
	std::map<nv::vec2i, GLuint>::iterator it = g_drawIndexMap.find(index);
	if (it == g_drawIndexMap.end())
	{
		GLuint newIndex = glGenLists(1);;
		glNewList(newIndex, GL_COMPILE);
		
		/*
			glBegin(GL_TRIANGLE_STRIP);
		   nv::vec3f point1 = 	nv::vec3f(-10.0046988,-13.1721592	,-43.1556740);
		   nv::vec3f point2 = 	nv::vec3f(	-9.55374241,-13.2565308,-43.3332710);
		   nv::vec3f eocCameraPos = nv::vec3f(15.6550808,-12.9208326,-49.8526649 );
		   const float farDis = 150;
		   nv::vec3f tex1 = point1 + farDis* normalize(point1 - eocCameraPos);
		    nv::vec3f tex2 = point2 + farDis* normalize(point2 - eocCameraPos);
			glVertex3fv((float*)&tex1);
			glVertex3fv((float*)&tex2);
			glVertex3fv((float*)&point1);
			glVertex3fv((float*)&point2);
			*/
			
		glBegin(GL_POINTS);
		//float x = 440.5, y = 718.5;
		for (float x = 0.5; x < w; x++)
		{
			for (float y = 0.5; y < h; y++)
			{
				//change
				float beginX = -1.0f + 2.0*  x / (w);
				float beginY = -1.0f + 2.0 * y / (h);
				glVertex2f(beginX, beginY);
			}
		}
		glEnd();
		glEndList();
		g_drawIndexMap.insert(pair<nv::vec2i, GLuint>(index, newIndex));
		return newIndex;
	}
	else
		return it->second;
}
std::map<GLuint, GLuint> testMap;
void myGeometry::drawQuadMesh(glslShader& shader,int m_w,int m_h)
{
	nv::vec2i index = nv::vec2i(m_w, m_h);
	std::map<nv::vec2i, GLuint>::iterator it = g_drawIndexMap.find(index); 
	int renderListId;

	std::map<GLuint, GLuint>::iterator testIt = testMap.find(1);
	if (it == g_drawIndexMap.end())
	{
		renderListId = myGeometry::initImageMesh(m_w,m_h);
	}
	else
	{
		renderListId = it->second;
	}

	shader.begin();
	glEnable(GL_TEXTURE_2D);
	glCallList(renderListId);
	shader.end();
}
void myGeometry::drawTriangle(nv::vec3f newOrigin, glslShader& shader)
{
	CHECK_ERRORS();
	nv::vec3f point1 = nv::vec3f(-4.3123, -10.4843, -51.6457);
	nv::vec3f point2 = nv::vec3f(-4.3123, -18.6958, -51.6457);

	const float farDis = 150;

	nv::vec3f vec1 = point1 + farDis* normalize(point1 - newOrigin);
	nv::vec3f vec2 = point2 + farDis* normalize(point2 - newOrigin);
	CHECK_ERRORS();
	shader.begin();
	CHECK_ERRORS();
	glBegin(GL_TRIANGLE_STRIP);
	glVertex3fv(vec1);
	glVertex3fv(vec2);
	glVertex3fv(point1);
	glVertex3fv(point2);
	glEnd();
	CHECK_ERRORS();
	shader.end();
	CHECK_ERRORS();

}
void myGeometry::appendDynamicObject(myGeometry& dynamicObject)
{
	if (_pOffset != dynamicObject._pOffset ||
		_nOffset != dynamicObject._nOffset ||
		_tcOffset != dynamicObject._tcOffset ||
		_vtxSize != dynamicObject._vtxSize ||
		_posSize != dynamicObject._posSize ||
		_normalSize != dynamicObject._normalSize ||
		_tcSize != dynamicObject._tcSize)
	{
		printf("can not append dynamicObject, format not match!!\n");
		return;
	}
	_groupIndices.insert(_groupIndices.end(), dynamicObject._groupIndices.begin(), dynamicObject._groupIndices.end());
	_indices.insert(_indices.end(), dynamicObject._indices.begin(), dynamicObject._indices.end());
	
	_vertices.insert(_vertices.end(), dynamicObject._vertices.begin(), dynamicObject._vertices.end());
	_materials.insert(_materials.end(), dynamicObject._materials.begin(), dynamicObject._materials.end());

	_groupMaterial.insert(_groupMaterial.end(), dynamicObject._groupMaterial.begin(), dynamicObject._groupMaterial.end());

	_groupNum += dynamicObject._groupNum;

}






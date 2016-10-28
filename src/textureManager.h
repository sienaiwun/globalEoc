
#pragma once

#include <gl/glew.h>
#include <vector>
#include <map>
#include <string>

struct dimension
{
	int m_x, m_y;
	dimension():m_x(-1), m_y(-1){};
	dimension(int x, int y) :m_x(x), m_y(y){};
	int getWidth() const
	{
		return m_x;
	}
	int getHeight() const
	{
		return m_y;
	}
};
using namespace std;
class textureManager{
public:
	textureManager(char * texDir){ m_textDir = string(texDir); };
	~textureManager();
	int getTexId(const char * texPath);
	map<string, dimension>m_nameToDimention;
	map<string, GLuint> m_nameToTexId;
private:
	
	
	string m_textDir;

};
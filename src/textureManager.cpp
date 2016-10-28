#include "textureManager.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <GL\glew.h>
#include <gl\gl.h>                    
#include <gl\glu.h>                    
//#include <gl\glaux.h>
#include <crtdbg.h>
#include <algorithm>
#include <olectl.h>              
#include <math.h>     
#include <opencv2/opencv.hpp>


using namespace std;

#define MAX_TEXTURES 100                // 最大的纹理数目

textureManager::~textureManager()
{
	auto destroyFun = [](map< string, GLuint>::value_type it){glDeleteTextures(1, &(it.second)); };
	std::for_each(m_nameToTexId.begin(), m_nameToTexId.end(), destroyFun);
}
#define CHECK_ERRORS()         \
	do {                         \
	GLenum err = glGetError(); \
	if (err) {                                                       \
	printf( "GL Error %d at line %d of FILE %s\n", (int)err, __LINE__,__FILE__);       \
	exit(-1);                                                      \
					}                                                                \
					} while(0)
int BuildTexture(const char *szPathName, GLuint &texid)
{
	HDC      hdcTemp;                        // The DC To Hold Our Bitmap
	HBITMAP    hbmpTemp;                        // Holds The Bitmap Temporarily
	IPicture  *pPicture;                        // IPicture Interface
	OLECHAR    wszPath[MAX_PATH + 1];                  // Full Path To Picture (WCHAR)
	char    szPath[MAX_PATH + 1];                    // Full Path To Picture
	long    lWidth;                          // Width In Logical Units
	long    lHeight;                        // Height In Logical Units
	long    lWidthPixels;                      // Width In Pixels
	long    lHeightPixels;                      // Height In Pixels
	GLint    glMaxTexDim;                      // Holds Maximum Texture Size

	if (strstr(szPathName, "http://"))                  // If PathName Contains http:// Then...
	{
		strcpy(szPath, szPathName);                    // Append The PathName To szPath
	}
	else                                // Otherwise... We Are Loading From A File
	{
		GetCurrentDirectory(MAX_PATH, szPath);              // Get Our Working Directory
		strcat(szPath, "\\");                      // Append "\" After The Working Directory
		strcat(szPath, szPathName);                    // Append The PathName
	}

	MultiByteToWideChar(CP_ACP, 0, szPath, -1, wszPath, MAX_PATH);    // Convert From ASCII To Unicode
	HRESULT hr = OleLoadPicturePath(wszPath, 0, 0, 0, IID_IPicture, (void**)&pPicture);

	if (FAILED(hr))                            // If Loading Failed
		return FALSE;                          // Return False

	hdcTemp = CreateCompatibleDC(GetDC(0));                // Create The Windows Compatible Device Context
	if (!hdcTemp)                            // Did Creation Fail?
	{
		pPicture->Release();                      // Decrements IPicture Reference Count
		return FALSE;                          // Return False (Failure)
	}

	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &glMaxTexDim);          // Get Maximum Texture Size Supported

	pPicture->get_Width(&lWidth);                    // Get IPicture Width (Convert To Pixels)
	lWidthPixels = MulDiv(lWidth, GetDeviceCaps(hdcTemp, LOGPIXELSX), 2540);
	pPicture->get_Height(&lHeight);                    // Get IPicture Height (Convert To Pixels)
	lHeightPixels = MulDiv(lHeight, GetDeviceCaps(hdcTemp, LOGPIXELSY), 2540);

	// Resize Image To Closest Power Of Two
	if (lWidthPixels <= glMaxTexDim) // Is Image Width Less Than Or Equal To Cards Limit
		lWidthPixels = 1 << (int)floor((log((double)lWidthPixels) / log(2.0f)) + 0.5f);
	else // Otherwise Set Width To "Max Power Of Two" That The Card Can Handle
		lWidthPixels = glMaxTexDim;

	if (lHeightPixels <= glMaxTexDim) // Is Image Height Greater Than Cards Limit
		lHeightPixels = 1 << (int)floor((log((double)lHeightPixels) / log(2.0f)) + 0.5f);
	else // Otherwise Set Height To "Max Power Of Two" That The Card Can Handle
		lHeightPixels = glMaxTexDim;

	//  Create A Temporary Bitmap
	BITMAPINFO  bi = { 0 };                        // The Type Of Bitmap We Request
	DWORD    *pBits = 0;                        // Pointer To The Bitmap Bits

	bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);        // Set Structure Size
	bi.bmiHeader.biBitCount = 32;                  // 32 Bit
	bi.bmiHeader.biWidth = lWidthPixels;              // Power Of Two Width
	bi.bmiHeader.biHeight = lHeightPixels;            // Make Image Top Up (Positive Y-Axis)
	bi.bmiHeader.biCompression = BI_RGB;                // RGB Encoding
	bi.bmiHeader.biPlanes = 1;                  // 1 Bitplane

	//  Creating A Bitmap This Way Allows Us To Specify Color Depth And Gives Us Imediate Access To The Bits
	hbmpTemp = CreateDIBSection(hdcTemp, &bi, DIB_RGB_COLORS, (void**)&pBits, 0, 0);

	if (!hbmpTemp)                            // Did Creation Fail?
	{
		DeleteDC(hdcTemp);                        // Delete The Device Context
		pPicture->Release();                      // Decrements IPicture Reference Count
		return FALSE;                          // Return False (Failure)
	}

	SelectObject(hdcTemp, hbmpTemp);                  // Select Handle To Our Temp DC And Our Temp Bitmap Object

	// Render The IPicture On To The Bitmap
	pPicture->Render(hdcTemp, 0, 0, lWidthPixels, lHeightPixels, 0, lHeight, lWidth, -lHeight, 0);

	// Convert From BGR To RGB Format And Add An Alpha Value Of 255
	for (long i = 0; i < lWidthPixels * lHeightPixels; i++)        // Loop Through All Of The Pixels
	{
		BYTE* pPixel = (BYTE*)(&pBits[i]);              // Grab The Current Pixel
		BYTE temp = pPixel[0];                  // Store 1st Color In Temp Variable (Blue)
		pPixel[0] = pPixel[2];                  // Move Red Value To Correct Position (1st)
		pPixel[2] = temp;                      // Move Temp Value To Correct Blue Position (3rd)

		// This Will Make Any Black Pixels, Completely Transparent    (You Can Hardcode The Value If You Wish)
		if ((pPixel[0] == 0) && (pPixel[1] == 0) && (pPixel[2] == 0))      // Is Pixel Completely Black
			pPixel[3] = 0;                      // Set The Alpha Value To 0
		else                              // Otherwise
			pPixel[3] = 255;                      // Set The Alpha Value To 255
	}
	BYTE* pData = new BYTE[lWidthPixels*lHeightPixels * 3];
	for (long i = 0; i < lWidthPixels * lHeightPixels; i++)        // Loop Through All Of The Pixels
	{
		pData[3 * i] = pBits[4 * i];

	}
	glGenTextures(1, &texid);                      // Create The Texture

	// Typical Texture Generation Using Data From The Bitmap
	glBindTexture(GL_TEXTURE_2D, texid);                // Bind To The Texture ID
	//glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);    // (Modify This For The Type Of Filtering You Want)
	//glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR); // (Modify This For The Type Of Filtering You Want)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, lWidthPixels, lHeightPixels, 0, GL_RGBA, GL_UNSIGNED_BYTE, pBits);  // (Modify This If You Want Mipmaps)
	CHECK_ERRORS();
	//gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA16, lWidthPixels, lHeightPixels, GL_RGBA, GL_UNSIGNED_BYTE, pBits);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	DeleteObject(hbmpTemp);                        // Delete The Object
	DeleteDC(hdcTemp);                          // Delete The Device Context

	pPicture->Release();                        // Decrements IPicture Reference Count

	return TRUE;                            // Return True (All Good)

}

int textureManager::getTexId(const char * texPath)
{
	glBindTexture(GL_TEXTURE_2D, 0);
	if (m_nameToTexId.count(texPath))
		return m_nameToTexId[texPath];
	if (texPath[0] == '\0')
		return -1;
	string fullPath = m_textDir  + texPath;
	
	IplImage *Iface = cvLoadImage(fullPath.c_str());
	if (!Iface)
	{
		printf("file error: %s\n", fullPath.c_str());
		return -1;
	}

	GLuint texid;
	glGenTextures(1, &texid);
	glBindTexture(GL_TEXTURE_2D, texid);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	CHECK_ERRORS();
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, Iface->width, Iface->height, 0,
		GL_BGR, GL_UNSIGNED_BYTE, Iface->imageData);
	CHECK_ERRORS();
	//BuildTexture(fullPath.c_str(), texid);
	glBindTexture(GL_TEXTURE_2D, 0);
	m_nameToTexId.insert(pair<string,int>(texPath,texid));
	m_nameToDimention.insert(pair<string, dimension>(texPath, dimension(Iface->width, Iface->height)));
	return texid;
}

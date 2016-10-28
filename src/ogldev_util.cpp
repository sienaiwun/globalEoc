/*

	Copyright 2014 Etay Meiri

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <iostream>
#include <fstream>
#ifdef WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#include "ogldev_util.h"

bool ReadFile(const char* pFileName, string& outFile)
{
    ifstream f(pFileName);
    
    bool ret = false;
    
    if (f.is_open()) {
        string line;
        while (getline(f, line)) {
            outFile.append(line);
            outFile.append("\n");
        }
        
        f.close();
        
        ret = true;
    }
    else {
        OGLDEV_FILE_ERROR(pFileName);
    }
    
    return ret;
}


void OgldevFileError(const char* pFileName, uint line, const char* pFileError)
{
#ifdef WIN32
	char msg[1000];
	_snprintf_s(msg, sizeof(msg), "%s:%d: unable to open file `%s`", pFileName, line, pFileError);
    MessageBoxA(NULL, msg, NULL, 0);
#else
    fprintf(stderr, "%s:%d: unable to open file `%s`\n", pFileName, line, pFileError);
#endif    
}


long long GetCurrentTimeMillis()
{
#ifdef WIN32    
	return GetTickCount();
#else
	timeval t;
	gettimeofday(&t, NULL);

	long long ret = t.tv_sec * 1000 + t.tv_usec / 1000;
	return ret;
#endif    
}
//void SaveBMP(const char *, BYTE *, int, int);
/*void SaveBMP(const char *filename, unsigned char *pixBuf, int width, int height){

	FILE *pFile;
	fopen_s(&pFile, filename, "wb");
	if (pFile == NULL){
		printf("Save file to %s : failed !\n", filename);
		return;
	}
	//------����˳��---------------------
	BYTE *m_PixChanged = new BYTE[width* height * 3];
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int inx = 3 * (i * width + j);
			m_PixChanged[inx + 0] = pixBuf[inx + 2];
			m_PixChanged[inx + 1] = pixBuf[inx + 1];
			m_PixChanged[inx + 2] = pixBuf[inx + 0];
		}
	}
	//-------- Create a new file for writing -----------------
	//-------------------- λͼ��Ϣͷ -----------------------
	BITMAPINFOHEADER BMIH;                 //λͼ��Ϣͷ
	BMIH.biSize = sizeof(BITMAPINFOHEADER);//ָ������ṹ�ĳ��ȣ�40�ֽ�

	//ָ��ͼ��Ŀ�ȣ���λ�����ء�
	BMIH.biWidth = width;

	//	ָ��ͼ��ĸ߶ȣ���λ�����ء�
	BMIH.biHeight = height;

	//Ŀ���ͼ�豸�����Ĳ�������������Ϊ1�����ÿ��ǡ�
	BMIH.biPlanes = 1;

	//ָ����ʾ��ɫʱҪ�õ���λ�������õ�ֵΪ1(�ڰ׶�ɫͼ), 4(16ɫͼ), 8(256ɫ), 24(���ɫͼ)
	BMIH.biBitCount = 24;

	//ָ��λͼ�Ƿ�ѹ������Ч��ֵΪBI_RGB��BI_RLE8��BI_RLE4��BI_BITFIELDS��BI_RGB��ѹ��
	BMIH.biCompression = BI_RGB;

	//ָ��ʵ�ʵ�λͼ����ռ�õ��ֽ���
	BMIH.biSizeImage = width * height * 3;

	//-------------------- λͼ�ļ�ͷ -----------------------
	BITMAPFILEHEADER bmfh;

	//λͼ��Ϣͷ���ļ�ͷ��С
	int nBitsOffset = sizeof(BITMAPFILEHEADER) + BMIH.biSize;

	//BMPͼ�����ݵĴ�С
	LONG lImageSize = BMIH.biSizeImage;

	//BMPͼ���ļ��Ĵ�С
	LONG lFileSize = nBitsOffset + lImageSize;

	//λͼ��𣬸��ݲ�ͬ�Ĳ���ϵͳ����ͬ����Windows�У����ֶε�ֵ��Ϊ��BM��
	bmfh.bfType = 'B' + ('M' << 8);

	//BMPͼ�����ݵĵ�ַ
	bmfh.bfOffBits = nBitsOffset;

	//BMPͼ���ļ��Ĵ�С
	bmfh.bfSize = lFileSize;

	//������Ϊ0
	bmfh.bfReserved1 = bmfh.bfReserved2 = 0;

	//Write the bitmap file header
	//��λͼ�ļ�ͷд��pFile
	UINT nWrittenFileHeaderSize = fwrite(&bmfh, 1, sizeof(BITMAPFILEHEADER), pFile);

	//And then the bitmap info header
	//��λͼ��Ϣͷд��pFile
	UINT nWrittenInfoHeaderSize = fwrite(&BMIH, 1, sizeof(BITMAPINFOHEADER), pFile);

	//Finally, write the image data itself 
	//��λͼ�ļ�ͼ������д��pFile
	//-- the data represents our drawing

	UINT nWrittenDIBDataSize = fwrite(m_PixChanged, 1, lImageSize, pFile);

	fclose(pFile);

	delete[]m_PixChanged;
}
*/



#ifdef WIN32
/*float fmax(float a, float b)
{
    if (a > b)
        return a;
    else
        return b;
}*/
#endif
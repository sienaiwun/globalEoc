#version 430

layout(location = 0) in vec2 tc;

uniform sampler2D colorTex;

uniform sampler2D normalTex;
uniform sampler2D posTex;
uniform sampler2D edgeTex;

uniform mat4 MVP; // Projection * ModelView
uniform mat4 mvpInv;
uniform mat4 modeView;
uniform mat4 modeViewInv;

uniform vec2 resolution;
uniform vec3 cameraPos;
layout (location =  0) out vec4 color0;  //pos
layout (location =  1) out vec4 color1;  //pos



float getDepthRep(vec3 worldPos)
{
	vec4 tmp = modeView*vec4(worldPos,1);
	return 1.0/tmp.z;
}
float getDepthTx(vec2 tc)
{
   return getDepthRep(texture2D(posTex,tc).xyz);
}

#define FISTDIS 0.0005
vec2 fistDepth(float c,float  l,float r)
{
	vec2 result;
	if (abs(l+r-2*c)<FISTDIS)
		result.x = 0;
	else
		result.x = 1;
	if(c+FISTDIS<r)
		result.y = 1;
	else
		result.y = 0;
	return result;
}
vec4 firstDisconect(vec2 tc)
{
	vec2 step = 1.0/resolution;
    vec4 result;
	float c = getDepthTx(tc);
	float l = getDepthTx(tc-vec2(step.x,0));
	float r = getDepthTx(tc+vec2(step.x,0));
	float t = getDepthTx(tc+vec2(0,step.y));
	float b = getDepthTx(tc-vec2(0,step.y));
	result.xy = fistDepth(c,l,r);
	result.zw = fistDepth(c,t,b);
	return result;
}
bool isEdge(vec2 uv)
{
	return texture2D(edgeTex,uv).z>0.5;
}
//x ��ֵ˵���ұ߱��������Զ�����߽������ȷ��

//y ��ֵ˵������߱��±�����Զ�����߽����±�ȷ��

//y ֵ��x ����ֵ����xȷ��
 bool assignPosX(vec2 uv)
 {
	if(!isEdge(uv))
	{
		color0.x= texture2D(posTex,uv).w;
		color0.yzw =  texture2D(posTex,uv).xyz;
		return true;
	}
	return false;
 }
 bool assignPosY(vec2 uv)//��uv���ڵ�texture�㸳ֵ����ǰps���ڵ�����
 {
	if(!isEdge(uv))
	{
		color1.x = texture2D(posTex,uv).w;
		color1.yzw = texture2D(posTex,uv).xyz;
		return true;
	}
	return false;
 }
 void vertiAssign(vec2 sobalValue,vec2 sign,vec2 tc)// ���ﶼ�Ǵ����ϱ߽�
 {
	vec2 step = 1.0/resolution;
		
	if(abs(sobalValue.y/sobalValue.x)<1.0/1.732 )//���ˮƽ����ռ����������ֱ��
	{
		
		
		
			if(assignPosY(tc+vec2(sign.x*step.x,0)))//��ѡˮƽ
			{	
				
	
				return;
			}
			else if(assignPosY(tc+vec2(sign.x*step.x,-step.y)))//��ѡ�±�
			{
				
				return;
			}
			else if(assignPosY(tc+vec2(sign.x*step.x,step.y)))//��ѡˮƽ��
			{
			
				return;
			}

	}
	else if(abs(sobalValue.y/sobalValue.x)<1.0)//��ֱ��ˮƽ������࣬ˮƽռ��Ҫ
	{
		
		if(assignPosY(tc+vec2(sign.x*step.x,-step.y)))
		{
			
	
			return;
		}
		else if (assignPosY(tc+vec2( 0,-step.y)))//��ѡˮƽ
		{
			return;
		}
		else if (assignPosY(tc+vec2(sign.x*step.x,  0)))
		{
			return;
		} 
	}
	else if(abs(sobalValue.y/sobalValue.x)<1.732)//��ֱ��ˮƽ������࣬ˮƽռ��Ҫ
	{

		if(assignPosY(tc+vec2(sign.x*step.x,-step.y)))
		{
		
			return;
		}
		else if (assignPosY(tc+vec2(sign.x*step.x,  0)))
		{
			return;
		} 
		else if (assignPosY(tc+vec2( 0,-step.y)))//��ѡˮƽ
		{
			return;
		}
	}
	else if(abs(sobalValue.y/sobalValue.x)>1.732)//�����ֱ����ռ������ ˮƽ��
	{
	
		if(assignPosY(tc+vec2(0,-step.y)))
		{
		
	
			return;
		}
		else if (assignPosY(tc+vec2(sign.x*step.x,  -step.y)))
		{
			return;
		} 
		else if (assignPosY(tc+vec2( -sign.x*step.x,-step.y)))//��ѡˮƽ
		{
			return;
		}
	}
 }
 void horiAssign(vec2 sobalValue,vec2 sign,vec2 tc)
 {
	vec2 step = 1.0/resolution;
	if(abs(sobalValue.y/sobalValue.x)>1.732 )//�����ֱ����ռ������
	{
			
		
		if(assignPosX(tc+vec2(0,sign.y*step.y)))//��ѡ�����
		{	
			return;
		}
		else if(assignPosX(tc+vec2(-step.x,sign.y*step.y)))//��ѡ���
		{
			return;
		}
		else if(assignPosX(tc+vec2(step.x,sign.y*step.y)))//ѡ�ұ�
		{
			return;
		}
	}
	else if(abs(sobalValue.y/sobalValue.x)>1.0)//��ֱ��ˮƽ������࣬��ֱռ��Ҫ
	{
		if(assignPosX(tc+vec2(-step.x,sign.y*step.y)))//��ѡ����
		{
			return;
		}
		else if (assignPosX(tc+vec2(0 , sign.y*step.y)))//��ѡ����
		{
			return;
		}
		else if (assignPosX(tc+vec2(-step.x, 0)))
		{
			return;
		}
	}
	else if(abs(sobalValue.y/sobalValue.x)>1.0/1.732)//��ֱ��ˮƽ������࣬ˮƽռ��Ҫ
	{
		if(assignPosX(tc+vec2(-step.x,sign.y*step.y)))
		{
			return;
		}
		else if (assignPosX(tc+vec2(-step.x ,0)))//��ѡˮƽ
		{
			return;
		}
		else if (assignPosX(tc+vec2(0, sign.y*step.y)))
		{
			return;
		}
	}
	else if(abs(sobalValue.y/sobalValue.x)< 1.0/1.732)//���ˮƽ����ռ������
	{

					
			if (assignPosX(tc+vec2(-step.x,0)))//��ѡ��
			{
				return;
			}
			else if (assignPosX(tc+vec2(-step.x,sign.y*step.y))) //��ѡ�������
			{
				return;
			}
			else if (assignPosX(tc+vec2(-step.x,-sign.y*step.y)))//��ѡ�����һ����
			{
				return;
			}
			
	}
 }
 
 //����ֱ�����Ͻ��и�ֵ

void main()
{
	vec2 sign;
	vec2 step = 1.0/resolution;
	if(isEdge(tc))
	{
		vec2 sobalValue =  texture2D(edgeTex,tc).xy;
		if(sobalValue.y>0)
		{
			sign.y = -1;	//�������ֵȡ����ֵ
		}
		else
		{
			sign.y = 1;		
		}
		if(sobalValue.x>0) //ȡ����ĵ�
		{
			sign.x = -1;
		}
		else
		{
			sign.x = 1;
		}
		if(sobalValue.x>0)
		{
			horiAssign(sobalValue,sign,tc);	
		}
		if(sobalValue.y >0)//
		{
		
			vertiAssign(sobalValue,sign,tc);
		}
	}
		
}
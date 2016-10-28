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
//x 正值说明右边比左边深度深（远），边界由左边确定

//y 正值说明上面边比下边深度深（远），边界由下边确定

//y 值比x 绝对值大又x确定
 bool assignPos(vec2 uv)
 {
	if(!isEdge(uv))
	{
		color0.x= 1;
		color0.yzw =  texture2D(posTex,uv).xyz;
		return true;
	}
	return false;
 }
void main()
{
	vec2 sign;
	vec2 step = 1.0/resolution;
	if(isEdge(tc))
	{
		vec2 sobalValue =  texture2D(edgeTex,tc).xy;
		if(sobalValue.y>0)
		{
			sign.y = -1;		
		}
		else
		{
			sign.y = 1;		
		}
		if(sobalValue.x>0)
		{
			sign.x = -1;
		}
		else
		{
			sign.x = 1;
		}
		if(sobalValue.x>0)
		{
				
		
			if(abs(sobalValue.y/sobalValue.x)>1.732 )
			{
				
				if(assignPos(tc+vec2(0,sign.y*step.y)))
				{	
					return;
				}
				else if(assignPos(tc+vec2(-step.x,sign.y*step.y)))
				{
					return;
				}
				else if(assignPos(tc+vec2(step.x,sign.y*step.y)))
				{
					return;
				}
			}
			else if(abs(sobalValue.y/sobalValue.x)>1.0)
			{
				if(assignPos(tc+vec2(-step.x,sign.y*step.y)))
				{
					return;
				}
				else if (assignPos(tc+vec2(0 , sign.y*step.y)))
				{
					return;
				}
				else if (assignPos(tc+vec2(-step.x, 0)))
				{
					return;
				}
			}
			else if(abs(sobalValue.y/sobalValue.x)>1.0/1.732)
			{
				if(assignPos(tc+vec2(-step.x,sign.y*step.y)))
				{
					return;
				}
				else if (assignPos(tc+vec2(-step.x ,0)))
				{
					return;
				}
				else if (assignPos(tc+vec2(0,  sign.y*step.y)))
				{
					return;
				}
			}
			else
			{

					
				 if (assignPos(tc+vec2(-step.x,0)))
				 {
					return;
				 }
				 else if (assignPos(tc+vec2(-step.x,sign.y*step.y)))
				 {
					return;
				 }
				  else if (assignPos(tc+vec2(-step.x,-sign.y*step.y)))
				 {
					return;
				 }
			
			}
			
		}
	}
		
}
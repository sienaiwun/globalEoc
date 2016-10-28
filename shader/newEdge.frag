#version 430

in vec3 worldPosP;
in vec3 worldNormal;
in vec2 tct;

uniform sampler2D colorTex;

uniform sampler2D normalTex;
uniform sampler2D posTex;


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
vec4 firstDisconect(vec2 tc,vec3 worldPos)
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

vec2 getTc(vec3 worldPos)
{
	vec4 temp = MVP*vec4(worldPos,1);
	return temp.xy/temp.w *0.5+0.5;
}
float disMinus(vec2 pos_uv,vec2 neg_uv)
{
	return length( texture2D(posTex,pos_uv).xyz-cameraPos)-length( texture2D(posTex,neg_uv).xyz-cameraPos);
}
bool isHeiher(vec2 uv,vec3 originPos)
{
	vec3 texPos = texture2D(posTex,uv).xyz;
	return (length(originPos-cameraPos)-length(texPos-cameraPos))>5.004;
}
bool leftIsHigher(vec2 tc,vec3 originPos)
{
	vec2 step = 1.0/resolution;
	bool l = isHeiher(tc+vec2(-step.x,0),originPos);
	bool lt =  isHeiher(tc+vec2(-step.x,step.y),originPos);
	bool lb =  isHeiher(tc+vec2(-step.x,-step.y),originPos);
	return l||lt||lb;
}
float sobalHori(vec2 tc)
{
	vec2 step = 1.0/resolution;
	float sobelValue = 1.414 * disMinus(tc + vec2(step.x,0),tc + vec2(-step.x,0)) + disMinus(tc + vec2(step.x,step.y),tc + vec2(-step.x,step.y)) + disMinus(tc + vec2(step.x,-step.y),tc + vec2(-step.x,-step.y));
	return sobelValue;
}
float sobelVertical(vec2 tc)
{
	vec2 step = 1.0/resolution;
	float sobelValue = 1.414 * disMinus(tc + vec2(0,step.y),tc + vec2(0,-step.y)) + disMinus(tc + vec2(step.x,step.y),tc + vec2(step.x,-step.y)) + disMinus(tc + vec2(-step.x,step.y),tc + vec2(-step.x,step.y));
	return sobelValue;
}
void main()
{
	
	vec2 ndc = (gl_FragCoord.xy)/resolution;
	vec3 orginPos = texture2D(posTex,ndc).xyz;
	
	//边界是在法线垂直于实现的的地方
	vec3 norm   = normalize( (mat3(modeViewInv)* worldNormal));
	if(abs(norm.z)>0.6)
		discard;

	if(length(orginPos-cameraPos)-length(worldPosP-cameraPos)>0.004)
		;
	else
		discard;
	if(length(orginPos-worldPosP)>8.004)
		;
	else
		discard;
	//vec2 stept = 1.0/resolution;
	//color0.xyzw = firstDisconect(ndc,worldPos);  //tc get by mvp
	
	color0.x =sobalHori(ndc);
	color0.y =sobelVertical(ndc);
	color0.z = 1;
	/*
	if(sobelVertical(ndc))
	{
		color0.x = 1;
		color0.yzw = worldPosP;	

	}*/
	return;
	color0.xy = (ndc+vec2(-step.x,0))*1024;
	
	vec3 neoPos =  texture2D(posTex,ndc+vec2(-step.x,0)).xyz;

	color0.xyz = neoPos;
	if(isHeiher(ndc+vec2(-step.x,0),orginPos))
		color0.y = 1.1;
//	color0.y = length(neoPos-cameraPos)-length(orginPos-cameraPos);
	return;
	
		
}
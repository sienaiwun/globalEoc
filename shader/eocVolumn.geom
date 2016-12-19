#version 430

layout (points) in ;
layout (triangle_strip, max_vertices = 16) out;

uniform mat4 MVP; // Projection * ModelView
uniform vec3 eocCameraPos;

in vec3 worldPos[];

out vec3 pWorldPos;


uniform sampler2D edgeTex;  //edgetex 里面第一位写是否是edge,234位写的是位置信息// 分两次渲染，分布传入x方向的prog
uniform sampler2D posTex;
uniform vec2 resolution;
uniform int isVertical;


void emitVertex(vec3 pos)
{
	gl_Position =  MVP * vec4(pos,1.0);
	pWorldPos  = pos;
	EmitVertex();
}
void emitTcVertex(vec2 tc,vec3 posValue)
{
	gl_Position.xy = tc;
	gl_Position.zw = vec2(0,1);
	pWorldPos  =	posValue;
	EmitVertex();
}

bool isTcOnVerticalEdge(vec2 tc)
{
	return  texture2D(edgeTex,tc).x >0.5f;
}
bool isTcOnHorizontalEdge(vec2 tc)
{
	return  texture2D(edgeTex,tc).x >0.5f;
}
bool neighborSearchValid(vec2 tc,float length)
{
	vec2 step = 1.0/resolution;
	for (float x = -length;x<=length;x++)
	{
		for (float y = -length;y<=length;y++)
		{
			 if(isTcOnVerticalEdge(tc+vec2(step.x*x,step.y*y)))
				return true;
		}
	}
	return false;

}
bool VerticalEdgeTest(vec3 worldPos,float length)
{
	vec4 temp = MVP * vec4(worldPos,1.0);
	vec2 tc = temp.xy/temp.w *0.5+0.5;
	return neighborSearchValid(tc,length);
}
bool isNeighborValid(vec3 orginPos,vec2 targetTc)
{
	if(isTcOnVerticalEdge(targetTc))
	{
		vec3 targetPos = texture2D(posTex,targetTc).xyz;
		//if(length(targetPos-orginPos)<5.0)
			return  true;
		//else
		//	return false;
	}
	else
		return false;
}
void emitVolume(vec3 point1, vec2 targetTc)
{
	if(!isTcOnVerticalEdge(targetTc))
		return;
	vec3 point2 = texture2D(edgeTex,targetTc).yzw;
	
	const float farDis = 150;
	vec3 tex1 = point1 + farDis* normalize(point1 - eocCameraPos);
	vec3 tex2 = point2 + farDis* normalize(point2 - eocCameraPos);
			emitVertex(tex1);
			emitVertex(tex2);
			emitVertex(point1);
			emitVertex(point2);
			EndPrimitive();
}

void emitHoriVolume(vec3 point1,vec2 targetTc)
{
	if(!isTcOnHorizontalEdge(targetTc))
		return;
	vec3 point2 = texture2D(edgeTex,targetTc).yzw;
	const float farDis = 150;
	vec3 tex1 = point1 + farDis* normalize(point1 - eocCameraPos);
	vec3 tex2 = point2 + farDis* normalize(point2 - eocCameraPos);
	emitVertex(tex1);
	emitVertex(tex2);
	emitVertex(point1);
	emitVertex(point2);
	EndPrimitive();
}
void main()
{	
	
	vec2 tc = worldPos[0].xy*0.5+0.5;
	if(isTcOnVerticalEdge(tc))
	{
		vec3 samplePos = texture2D(edgeTex,tc).yzw;
		vec2 step = 1.0/resolution;
		vec2 bot = tc + vec2(0,-step.y);//down
		emitVolume(samplePos,bot); 

		vec2 botLeft = tc + vec2(-step.x,-step.y); //left down
		emitVolume(samplePos,botLeft);

		vec2 botRight = tc + vec2(step.x,-step.y);//right down
		emitVolume(samplePos,botRight);
		
		vec2 right = tc + vec2(step.x,0); //right 
		emitVolume(samplePos,right);
	}
	
	/*
	
	//debug pixel
	vec3 outValue;
	vec2 step = 1.0/resolution;
	outValue.xy = step;
	vec3 samplePos = texture2D(edgeTex,tc).yzw;
	vec2 right = tc + vec2(step.x,0); //right 
	vec3 samplePos1 = texture2D(edgeTex,right).yzw;
	outValue =  texture2D(edgeTex,right).xyz;
	if(!isTcOnVerticalEdge(right))
		outValue = vec3(1,1,1);
	else
		outValue = vec3(2,2,1);
	emitTcVertex(vec2(-1,-1),outValue);
	emitTcVertex(vec2(-1,1),outValue);
	emitTcVertex(vec2(1,-1),outValue);
	emitTcVertex(vec2(1,1),outValue);
	EndPrimitive();
	return;
	*/
	/*origin method
    int isEdgeNum = 0;
	vec3 point1,point2,tex1,tex2;
	for(int i = 0;i<gl_in.length();i++)
	{
		int vIsEdgeI = int(VerticalEdgeTest( worldPos[i],1.5));
		isEdgeNum +=vIsEdgeI;
		
		if((isEdgeNum==1) && (vIsEdgeI == 1))
		{
			point1 =  worldPos[i];
		}
		else if((isEdgeNum==2 )&& (vIsEdgeI == 1))
		{
			point2 =  worldPos[i];
		}
	}
	if(isEdgeNum==2)
	{
		vec3 midPoint = (point1+point2)/2;
		if(VerticalEdgeTest(midPoint,2))
		{
			
			const float farDis = 150;
		  
			tex1 = point1 + farDis* normalize(point1 - eocCameraPos);
			tex2 = point2 + farDis* normalize(point2 - eocCameraPos);
			
			emitVertex(tex1);
			emitVertex(tex2);
			emitVertex(point1);
			emitVertex(point2);
			EndPrimitive();
		}
		
	}

	*/

}
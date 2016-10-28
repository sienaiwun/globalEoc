#version 430
uniform mat4 MVP; // Projection * ModelView
uniform mat4 modeViewInv;
uniform mat4 mvpInv;
uniform mat4 projection;
//in int gl_VertexID;
layout (location = 0) in vec3 VertexPosition;
layout (location = 1) in vec3 VertexNormal;
layout (location = 2) in vec2 VertexTc;

out vec3 worldPosP;
out vec3 worldNormal;
out vec2 tct;
//out int out_vertexId;
vec2 TransformViewToProjection(vec2 v)
{ 
	return (mat2(projection)* v);; 
}
vec2 offset(vec3 pos,vec3 normal)
{	
	vec3 norm   = normalize( (mat3(modeViewInv)* normal));
	if(abs(norm.z)>0.6)
		return vec2(0,0);
	vec4 temp1 = MVP * vec4(pos+normal ,1.0);
	vec4 temp2 = MVP * vec4(pos ,1.0);
	vec2 realPlace = temp2.xy/temp2.w;
	
	vec2 off = temp1.xy/temp1.w - realPlace;
	if(abs(realPlace.x)>1||abs(realPlace.y)>1)
		 return off;
	if(abs(off.x)>abs(off.y))
		return off/abs(off.x);
	else
		return off/abs(off.y);
}
void main()
{
	const float _Outline = 0.045;

	worldNormal = normalize(VertexNormal);
	tct = vec2(VertexTc.x,VertexTc.y);
	gl_Position = MVP * vec4(VertexPosition ,1.0);
	gl_Position/=gl_Position.w;
	gl_Position.xy += offset(VertexPosition,worldNormal)*1.0/1024*1;

	vec4 temp = mvpInv* gl_Position;
	worldPosP = temp.xyz/temp.w;
	gl_Position = MVP * vec4(worldPosP ,1.0);
	//vec3 norm   =  (mat3(modeViewInv)* worldNormal);
	//vec2 offset = TransformViewToProjection(norm.xy);
	//gl_Position.xy += offset * gl_Position.z * _Outline;

	//gl_Position = MVP * vec4(VertexPosition+worldNormal*0.05 ,1.0);

}
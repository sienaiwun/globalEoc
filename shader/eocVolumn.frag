#version 430
in vec3 pWorldPos;
in vec3 pdebug;

layout( location = 0 ) out vec4 FragColor0;
uniform mat4 MVP; // Projection * ModelView
uniform vec3 cameraPos;
uniform sampler2D edgeTex;
uniform sampler2D posTex;
uniform vec2 resolution;


bool isVerticalEdge(vec2 tc, vec3 worldPos)
{
    bvec2 test= greaterThan(texture2D(edgeTex,tc).xy , vec2(0.5f,0.5f));
	bool isEdge = all(test);
		
		
	if(isEdge)
	{
		
		if(length(texture2D(posTex,tc).xyz-worldPos)<0.25)
			return true ;
		else
			return false;
		
	}
	else 
		return false;
}


bool VerticalEdgeTest(vec2 tc,vec3 worldPos)
{
	vec2 step = 1.0/resolution;
   	bool c = isVerticalEdge(tc,worldPos);
	bool l = isVerticalEdge(tc-vec2(step.x,0),worldPos);
	bool r = isVerticalEdge(tc+vec2(step.x,0),worldPos);
	bool t = isVerticalEdge(tc+vec2(0,step.y),worldPos);
	bool b = isVerticalEdge(tc-vec2(0,step.y),worldPos);
	//return c||l||r||t||b;

	bool tr = isVerticalEdge(tc+vec2(step.x,step.y),worldPos);
	bool bl = isVerticalEdge(tc-vec2(step.x,step.y),worldPos);
	bool tl = isVerticalEdge(tc+vec2(step.x,-step.y),worldPos);
	bool br = isVerticalEdge(tc+vec2(-step.x,step.y),worldPos);

	return c||l||r||t||b||tr||bl||tl||br;
}

bool isTcOnVerticalEdge(vec2 tc)
{
	return   texture2D(edgeTex,tc).x >0.5f;
}
vec2 inToNdcPos(vec3 pos)
{
	vec4 temp = MVP* vec4(pos,1);
	return temp.xy/temp.w * 0.5+vec2(0.5,0.5);
}
void main()
{

	vec2 ndc = (gl_FragCoord.xy)/resolution;
/*	
	vec2 tc = pWorldPos.xy*0.5+0.5;
	vec3 edgePos = texture2D(edgeTex,tc).yzw;
	FragColor0.xyz = vec3(edgePos);
	vec2 step = 1.0/resolution;
	vec2 bot = tc + vec2(0,-step.y);
	FragColor0.x = float(isTcOnVerticalEdge(bot));
	vec2 botLeft = tc + vec2(-step.x,-step.y);
		FragColor0.y = float(isTcOnVerticalEdge(botLeft));

		vec2 botRight = tc + vec2(step.x,-step.y);
		FragColor0.z = float(isTcOnVerticalEdge(botRight));
		
	vec2 right = tc + vec2(step.x,0);
	FragColor0.w = float(isTcOnVerticalEdge(bot));
	FragColor0.xyz =  texture2D(posTex,botLeft).xyz;
	FragColor0.xy = inToNdcPos(texture2D(edgeTex,bot).yzw)*1024.0;
	FragColor0.zw = bot*1024;
	return;
	*/
	
	vec3 gBufferPos = texture2D(posTex,ndc).xyz;
	if(length(gBufferPos-cameraPos)>length(pWorldPos-cameraPos))
	{
		discard;
	}
	if(pWorldPos.x < -26.5745)
	{
		discard;
	}
	float ka = 0.2;
	FragColor0.xyz = vec3(1,1,1);
	return;
	
	/*
	vec3 outColor;
	if(VerticalEdgeTest(ndc,worldPos))
		outColor = vec3(1,1,0);
	else
		outColor = vec3(0,1,0);
	vec2 value = texture2D(edgeTex,ndc).xy;
	 bvec2 test= greaterThan(value , vec2(0.5f,0.5f));
	 bool isEdge = all(test);
	// vec3 VertexPosition = 	vec3(-4.3123, -10.4843, -51.6457);
	//vec3 VertexPosition = vec3(-4.3123, -18.6958, -51.6457);
	//vec3 VertexPosition =	 vec3(-142.864105,42.3037643,-28.9120674);
	vec3 VertexPosition=    vec3(-152.217331 ,-24.6246376 ,-27.3773918);

	vec4 temp = MVP * vec4(VertexPosition,1.0);
	vec2 testTc = temp.xy/temp.w *0.5+0.5;
	
	vec2 step = 1.0/resolution;
   	bool c = isVerticalEdge(testTc,VertexPosition);
	bool l = isVerticalEdge(testTc-vec2(step.x,0),VertexPosition);
	bool r = isVerticalEdge(testTc+vec2(step.x,0),VertexPosition);
	bool t = isVerticalEdge(testTc+vec2(0,step.y),VertexPosition);
	bool b = isVerticalEdge(testTc-vec2(0,step.y),VertexPosition);
	bool tr = isVerticalEdge(testTc+vec2(step.x,step.y),VertexPosition);
	bool bl = isVerticalEdge(testTc-vec2(step.x,step.y),VertexPosition);
	bool tl = isVerticalEdge(testTc+vec2(step.x,-step.y),VertexPosition);
	bool br = isVerticalEdge(testTc+vec2(-step.x,step.y),VertexPosition);
	float fc = float(c);
	float fl = float(l);
	float fr = float(r);
	float ft = float(t);
	float fbl = float(bl);
	vec3 posT = texture2D(posTex,testTc+vec2(0,step.y)).xyz;
	//FragColor0.xy = testTc*1024;
	//return;
	if(VerticalEdgeTest(testTc,VertexPosition))
		FragColor0.xyz = vec3(1,0,0);
	else
		FragColor0.xyz = vec3(0,1,1);
		*/

	
}

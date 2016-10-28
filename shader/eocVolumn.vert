#version 430
uniform mat4 MVP; // Projection * ModelView

layout (location = 0) in vec3 VertexPosition;

out vec3 worldPos;

uniform sampler2D edgeTex;
uniform sampler2D posTex;
uniform vec2 resolution;




bool isVerticalEdge(vec2 tc, vec3 worldPos)
{
    bvec2 test= greaterThan(texture2D(edgeTex,tc).xy , vec2(0.5f,0.5f));
	bool isEdge = all(test);
		
	
	if(isEdge)
	{
		
		//if(length(texture2D(posTex,tc).xyz-worldPos)<0.25)
			return true ;
		//else
		//	return false;
		
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

void main()
{
	worldPos = VertexPosition;
	gl_Position =  vec4(VertexPosition,1.0);
	
}
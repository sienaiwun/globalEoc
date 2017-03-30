
uniform mat4 MVP; // Projection * ModelView
//in int gl_VertexID;
varying vec3 normal;
//out int out_vertexId;
void main()
{
	vec4 worldPos = gl_Vertex;
	normal = gl_Normal;
	gl_Position = MVP * vec4(worldPos.xyz,1.0);
//	out_vertexId = VertexTc;
}
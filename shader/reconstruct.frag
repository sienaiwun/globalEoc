


uniform sampler2D objectTex;
uniform vec3 objectDiffuseColor;
uniform mat4 modelView; //  ModelView

uniform vec3 cameraPos;
uniform vec3 lightPos;
uniform int hasTex;
uniform int objectId;
uniform float reflectFactor;

varying vec3 normal;

void main()
{
	vec3 N = (normal.xyz);
	gl_FragColor .xyz = N;
	gl_FragColor .w= 1;
	/*
	vec4 test;
	#define N 100000
	for(int i =0;i<N;i++)
	{
		test.xyz += diffuse*(kd+ka)+vec3(0.2,0.2,0.2);
	}
	gl_FragColor.xyz =test.xyz/N;*/
}

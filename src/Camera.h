#ifndef _CAMERA_H
#define _CAMERA_H
#include<stdio.h>
#include <string>
#include <nvMath.h>
// This is our basic 3D point/vector class
class CVector3
{
public:

	// A default constructor
	CVector3() {}

	// This is our constructor that allows us to initialize our data upon creating an instance
	CVector3(float X, float Y, float Z)
	{
		x = X; y = Y; z = Z;
	}

	// Here we overload the + operator so we can add vectors together 
	CVector3 operator+(CVector3 vVector)
	{
		// Return the added vectors result.
		return CVector3(vVector.x + x, vVector.y + y, vVector.z + z);
	}

	// Here we overload the - operator so we can subtract vectors 
	CVector3 operator-(CVector3 vVector)
	{
		// Return the subtracted vectors result
		return CVector3(x - vVector.x, y - vVector.y, z - vVector.z);
	}

	// Here we overload the * operator so we can multiply by scalars
	CVector3 operator*(float num)
	{
		// Return the scaled vector
		return CVector3(x * num, y * num, z * num);
	}

	// Here we overload the / operator so we can divide by a scalar
	CVector3 operator/(float num)
	{
		// Return the scale vector
		return CVector3(x / num, y / num, z / num);
	}

	float x, y, z;
};

// This is our camera class
class Camera {
public:
	// Our camera constructor
	Camera();

	Camera(CVector3, CVector3, CVector3);

	void setPos(nv::vec3f newOrigin, nv::vec3f focus)
	{
		m_vPosition = CVector3(newOrigin.x, newOrigin.y, newOrigin.z);
		m_vView = CVector3(focus.x, focus.y, focus.z);
	}
	// These are are data access functions for our camera's private data
	CVector3 Position() { return m_vPosition; }
	CVector3 View()		{ return m_vView; }
	CVector3 UpVector() { return m_vUpVector; }
	CVector3 Strafe()	{ return m_vStrafe; }

	// This changes the position, view, and up vector of the camera.
	// This is primarily used for initialization
	void PositionCamera(float positionX, float positionY, float positionZ,
		float viewX, float viewY, float viewZ,
		float upVectorX, float upVectorY, float upVectorZ);

	// This rotates the camera's view around the position depending on the values passed in.
	void RotateView(float angle, float X, float Y, float Z);

	// This moves the camera's view by the mouse movements (First person view)
	void SetViewByMouse();

	// This rotates the camera around a point (I.E. your character).
	void RotateAroundPoint(CVector3 vCenter, float X, float Y, float Z);

	// This strafes the camera left or right depending on the speed (+/-) 
	void StrafeCamera(float speed);

	// This will move the camera forward or backward depending on the speed
	void MoveCamera(float speed);

	// This checks for keyboard movement
	void CheckForMovement();

	// This updates the camera's view and other data (Should be called each frame)
	void Update();

	// This uses gluLookAt() to tell OpenGL where to look
	void Look();

	// This returns the inverse to the current modelview matrix in OpenGL
	void GetInverseMatrix(float mCameraInverse[16]);
	

	void camLook();
	void cameraControl();

	void printToFile(std::string fileName) ;
	void loadToFIle(const char * fileName) ;
	inline float * getProjection() const
	{
		return (float*)m_projMat.get_value();
	}
	inline  float * getModelViewMat() const
	{
		return (float*)m_modelView.get_value();
	}
	inline  float * getMvpMat() const
	{
		return (float*)m_mvpMat.get_value();
	}
	/*
	inline float* getInvMvp() const
	{
		//这样写是错误的，空指正
		return (float*)inverse(m_mvpMat).get_value();
	}
	inline  float * getModelViewInvMat() const
	{
		//这样写是错误的，空指正
		return (float*)inverse(m_modelView).get_value();
	}
	*/
	inline nv::vec3f getCameraPos() const
	{
		return nv::vec3f(m_vPosition.x, m_vPosition.y, m_vPosition.z);
	}
	inline nv::vec2f getImageMin() const
	{
		return m_framebbmin;
	}
	inline nv::vec2f getImageMax() const
	{
		return m_framebbmax;
	}
	inline nv::vec3f getUpND() const
	{
		return m_upD;
	}
	inline nv::vec3f getDeepND() const
	{
		return m_deepD;
	}
	inline nv::vec3f getRightND() const
	{
		return m_rightD;
	}
	void moveTo(float toOrigin, float toFocus);
	
private:
	// The camera's position
	CVector3 m_vPosition;
	// The camera's view
	CVector3 m_vView;
	// The camera's up vector
	CVector3 m_vUpVector;
	// The camera's strafe vector
	CVector3 m_vStrafe;
	nv::vec3f m_upD,m_rightD,m_deepD;
	nv::matrix4f m_modelView;
	nv::matrix4f m_mvpMat;
	nv::matrix4f m_projMat;
	nv::vec2f m_framebbmin, m_framebbmax;
	float old_x, old_y;
};


// This makes sure we only draw at a specified frame rate
bool AnimateNextFrame(int desiredFrameRate);


#endif


/////////////////////////////////////////////////////////////////////////////////
//
// * QUICK NOTES * 
//
// Nothing new was added to this file for our current tutorial.
// 
// 
// ?000-2005 GameTutorials
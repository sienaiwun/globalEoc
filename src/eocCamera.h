#include "Camera.h"
#ifndef  EOCCAMERA_H
#define EOCCAMERA_H
typedef enum 
{
	is_Right = 1,
	is_Top,
} EocType;
class EocCamera
{
public:
	EocCamera() :m_pOriginCamera(NULL)
	{

	};

	EocCamera(EocType type, float toOrigin, float toFocus) : m_type(type), m_toFucusDis(toFocus), m_toOriginDis(toOrigin)
	{}
	~EocCamera() = default;
	inline void setOriginCamera(Camera * pCamera)
	{
		m_pOriginCamera = pCamera;
	}
	inline Camera* getEocCameraP() 
	{
		return &m_eocCamera;
	}
	inline void addToOrigin(const float dis)
	{
		m_toOriginDis += dis;
		m_toOriginDis = std::max<float>(0.0, m_toOriginDis);
	}
	void Look();
private:
	Camera * m_pOriginCamera;
	Camera m_eocCamera;
	EocType m_type;
	float m_toFucusDis, m_toOriginDis;
};
#endif
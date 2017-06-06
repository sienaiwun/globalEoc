#include "eocCamera.h"
#include <assert.h>

void EocCamera::Look()
{
	assert(m_type == is_Right || m_type == is_Top);
	assert(m_pOriginCamera != NULL);
	nv::vec3f vecD;
	switch (m_type)
	{
	case is_Right:
		vecD  = m_pOriginCamera->getRightND();
		break;
	case is_Top:
		vecD = m_pOriginCamera->getUpND();
		break;
	default:
		break;
	}
	nv::vec3f newOrigin = m_pOriginCamera->getCameraPos() + m_toOriginDis*vecD;
	nv::vec3f newFocus = newOrigin + m_toFucusDis*m_pOriginCamera->getDeepND();
	m_eocCamera.setPos(newOrigin, newFocus);
	m_eocCamera.Look();
}


/*
void EocCamera::Look()
{
assert(m_type == is_Right || m_type == is_Top);
assert(m_pOriginCamera != NULL);
nv::vec3f vecD;
switch (m_type)
{
case is_Right:
vecD  = m_pOriginCamera->getRightND();
printf("is_Right\n");
break;
case is_Top:
vecD = m_pOriginCamera->getUpND();
printf("is_Top\n");
break;
default:
break;
}
nv::vec3f newOrigin = m_pOriginCamera->getCameraPos() + 0.5*m_toOriginDis*vecD;
newOrigin = m_pOriginCamera->getCameraPos() + 1*m_toOriginDis*m_pOriginCamera->getRightND() +1*m_toOriginDis*m_pOriginCamera->getUpND();
nv::vec3f newFocus = newOrigin + m_toFucusDis*m_pOriginCamera->getDeepND();
newFocus = newOrigin + nv::normalize(newFocus - newOrigin);
printf("newOrigin:(%f,%f,%f)\n",newOrigin.x,newOrigin.y,newOrigin.z);
printf("newFocus:(%f,%f,%f)\n", newFocus.x, newFocus.y, newFocus.z);
fflush(stdout);
m_eocCamera.setPos(newOrigin, newFocus);
m_eocCamera.Look();
}
*/
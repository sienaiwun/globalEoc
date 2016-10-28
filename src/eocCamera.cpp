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
	nv::vec3f newFocus = m_pOriginCamera->getCameraPos() + m_toFucusDis*m_pOriginCamera->getDeepND();
	m_eocCamera.setPos(newOrigin, newFocus);
	m_eocCamera.Look();


}

// Opencv_Project_003.h : PROJECT_NAME ���� ���α׷��� ���� �� ��� �����Դϴ�.
//

#pragma once

#ifndef __AFXWIN_H__
	#error "PCH�� ���� �� ������ �����ϱ� ���� 'stdafx.h'�� �����մϴ�."
#endif

#include "resource.h"		// �� ��ȣ�Դϴ�.


// COpencv_Project_003App:
// �� Ŭ������ ������ ���ؼ��� Opencv_Project_003.cpp�� �����Ͻʽÿ�.
//

class COpencv_Project_003App : public CWinApp
{
public:
	COpencv_Project_003App();

// �������Դϴ�.
public:
	virtual BOOL InitInstance();

// �����Դϴ�.

	DECLARE_MESSAGE_MAP()
};

extern COpencv_Project_003App theApp;
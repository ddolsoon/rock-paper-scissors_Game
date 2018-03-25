
// Opencv_Project_003.h : PROJECT_NAME 응용 프로그램에 대한 주 헤더 파일입니다.
//

#pragma once

#ifndef __AFXWIN_H__
	#error "PCH에 대해 이 파일을 포함하기 전에 'stdafx.h'를 포함합니다."
#endif

#include "resource.h"		// 주 기호입니다.


// COpencv_Project_003App:
// 이 클래스의 구현에 대해서는 Opencv_Project_003.cpp을 참조하십시오.
//

class COpencv_Project_003App : public CWinApp
{
public:
	COpencv_Project_003App();

// 재정의입니다.
public:
	virtual BOOL InitInstance();

// 구현입니다.

	DECLARE_MESSAGE_MAP()
};

extern COpencv_Project_003App theApp;
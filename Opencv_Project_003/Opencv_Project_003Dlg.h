
// Opencv_Project_003Dlg.h : 헤더 파일
//

#pragma once

#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv2/opencv.hpp"
#include  "CvvImage.h"
#include "afxwin.h"
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <time.h>
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;

#define _DEF_WEBCAM 1000

#define _DEF_COUNTDOWN 2000

template <typename T> string NumberToString(T Number);	//숫자를 스트링으로 바꿔주는 함수

Mat get_CrCb_HandMask3(Mat image); //손 영역 마스크 함수
Point getHandCenter(const Mat& mask, double& radius);	//손의 무게중심 함수
Mat detectAndDisplay(Mat frame);	//얼굴 검출 함수
int findBiggestContour(vector<vector<Point> > contours); //가장 큰 외곽선의 인덱스 반환 함수



// COpencv_Project_003Dlg 대화 상자
class COpencv_Project_003Dlg : public CDialogEx
{
// 생성입니다.
public:

	CFont label1, label2,label3,label4,label5;

	//비디오 캠 출력 변수
	IplImage * m_pImage;
	CvvImage m_cImage;
	cv::Mat m_mImage;

	cv::VideoCapture cam;
	
	//가위바위보 사용자/컴퓨터 이미지 변수
	IplImage * m_user_pImage;
	CvvImage m_user_cImage;
	cv::Mat m_user_mImage;
	IplImage * m_computer_pImage;
	CvvImage m_computer_cImage;
	cv::Mat m_computer_mImage;
	Mat window_image;

	//게임 카운트다운
	int countdown;

	COpencv_Project_003Dlg(CWnd* pParent = NULL);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
	enum { IDD = IDD_OPENCV_PROJECT_003_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:

	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnDestroy();
	CStatic m_video;
	CStatic m_user_image;
	CStatic m_computer_image;
	afx_msg BOOL OnEraseBkgnd(CDC* pDC);

	afx_msg void OnBnClickedButton();
};

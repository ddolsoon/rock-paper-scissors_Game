
// Opencv_Project_003Dlg.h : ��� ����
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

template <typename T> string NumberToString(T Number);	//���ڸ� ��Ʈ������ �ٲ��ִ� �Լ�

Mat get_CrCb_HandMask3(Mat image); //�� ���� ����ũ �Լ�
Point getHandCenter(const Mat& mask, double& radius);	//���� �����߽� �Լ�
Mat detectAndDisplay(Mat frame);	//�� ���� �Լ�
int findBiggestContour(vector<vector<Point> > contours); //���� ū �ܰ����� �ε��� ��ȯ �Լ�



// COpencv_Project_003Dlg ��ȭ ����
class COpencv_Project_003Dlg : public CDialogEx
{
// �����Դϴ�.
public:

	CFont label1, label2,label3,label4,label5;

	//���� ķ ��� ����
	IplImage * m_pImage;
	CvvImage m_cImage;
	cv::Mat m_mImage;

	cv::VideoCapture cam;
	
	//���������� �����/��ǻ�� �̹��� ����
	IplImage * m_user_pImage;
	CvvImage m_user_cImage;
	cv::Mat m_user_mImage;
	IplImage * m_computer_pImage;
	CvvImage m_computer_cImage;
	cv::Mat m_computer_mImage;
	Mat window_image;

	//���� ī��Ʈ�ٿ�
	int countdown;

	COpencv_Project_003Dlg(CWnd* pParent = NULL);	// ǥ�� �������Դϴ�.

// ��ȭ ���� �������Դϴ�.
	enum { IDD = IDD_OPENCV_PROJECT_003_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV �����Դϴ�.


// �����Դϴ�.
protected:
	HICON m_hIcon;

	// ������ �޽��� �� �Լ�
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


// Opencv_Project_003Dlg.cpp : 구현 파일
//

#include "stdafx.h"
#include "Opencv_Project_003.h"
#include "Opencv_Project_003Dlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//가위바위보 인식을 위한 변수
string face_cascade_name = "haarcascade_frontface.xml";
CascadeClassifier face_cascade;
int filenumber; // Number of file to be saved
string filename;
char cfileName[1024];
string sfileName = "";
Size outputSize = Size(960, 540);

int inputLayerSize = 7;
int outputLayerSize = 3;
int numSamples = 50;
vector<int> layerSizes = { inputLayerSize, outputLayerSize };
Ptr<ml::ANN_MLP> nnPtr = ml::ANN_MLP::create();

// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// COpencv_Project_003Dlg 대화 상자



COpencv_Project_003Dlg::COpencv_Project_003Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(COpencv_Project_003Dlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void COpencv_Project_003Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_PICTURE, m_video);
	DDX_Control(pDX, IDC_USER_INPUT, m_user_image);
	DDX_Control(pDX, IDC_COMPUTER_INPUT, m_computer_image);
}

BEGIN_MESSAGE_MAP(COpencv_Project_003Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_TIMER()
	ON_WM_DESTROY()
	ON_WM_ERASEBKGND()
	ON_BN_CLICKED(IDC_BUTTON, &COpencv_Project_003Dlg::OnBnClickedButton)
END_MESSAGE_MAP()


// COpencv_Project_003Dlg 메시지 처리기

BOOL COpencv_Project_003Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.

	//프로그램 레이블 폰트 처리
	label1.CreateFont(30, 15, 0, 0, 1000, 1, 0, 0, 0, OUT_DEFAULT_PRECIS, 0, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, _T("맑은고딕"));
	GetDlgItem(IDC_USER_LABEL)->SetFont(&label1);
	label2.CreateFont(30, 15, 0, 0, 1000, 1, 0, 0, 0, OUT_DEFAULT_PRECIS, 0, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, _T("맑은고딕"));
	GetDlgItem(IDC_COMPUTER_LABEL)->SetFont(&label2);
	label3.CreateFont(30, 15, 0, 0, 1000, 1, 0, 0, 0, OUT_DEFAULT_PRECIS, 0, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, _T("맑은고딕"));
	GetDlgItem(IDC_COUNT_LABEL)->SetFont(&label3);
	label4.CreateFont(100, 100, 0, 0, 1000, 1, 0, 0, 0, OUT_DEFAULT_PRECIS, 0, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, _T("맑은고딕"));
	GetDlgItem(IDC_COUNT_DISPLAY)->SetFont(&label4);
	label5.CreateFont(50, 25, 0, 0, 1000, 1, 0, 0, 0, OUT_DEFAULT_PRECIS, 0, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, _T("맑은고딕"));
	GetDlgItem(IDC_BUTTON)->SetFont(&label5);
	

	// Load the cascade (얼굴 검출 분류기를 load 한다.)
	if (!face_cascade.load(face_cascade_name)){
		printf("--(!)Error loading\n");
		return (-1);
	}

	//인공신경망(ANN) 분류기 초기화
	nnPtr->setLayerSizes(layerSizes);
	nnPtr->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);

	Mat samples(Size(inputLayerSize, numSamples * 3), CV_32F);
	//Mat image;
	Mat image;
	Mat labels(Size(outputLayerSize, numSamples * 3), CV_32F);

	//바위 학습
	Mat bawi_image;
	Mat bawi_handmask;
	int i = 0;
	for (; i < numSamples; i++)
	{
		string sfileName = "collect/bawi" + NumberToString(i + 1) + ".jpg";
		bawi_image = imread(sfileName,-1);
		//Mat bawi_handmask = get_CrCb_HandMask3(bawi_image);
		bawi_handmask = bawi_image;

		Moments m = cv::moments(bawi_handmask, false);
		double h[7];
		HuMoments(m, h);
		samples.at<float>(Point(0, i)) = (float)h[0];
		samples.at<float>(Point(1, i)) = (float)h[1];
		samples.at<float>(Point(2, i)) = (float)h[2];
		samples.at<float>(Point(3, i)) = (float)h[3];
		samples.at<float>(Point(4, i)) = (float)h[4];
		samples.at<float>(Point(5, i)) = (float)h[5];
		samples.at<float>(Point(6, i)) = (float)h[6];
	
		labels.at<float>(Point(0, i)) = 1.0f;
		labels.at<float>(Point(1, i)) = 0.0f;
		labels.at<float>(Point(2, i)) = 0.0f;
	}

	//가위 학습
	Mat gawi_image;
	Mat gawi_handmask;
	for (; i < numSamples * 2; i++)
	{
		string sfileName = "collect/gawi" + NumberToString(i + 1 - numSamples) + ".jpg";
		gawi_image = imread(sfileName,-1);
		//Mat gawi_handmask = get_CrCb_HandMask3(gawi_image);
		gawi_handmask = gawi_image;

		Moments m = cv::moments(gawi_handmask, false);
		double h[7];
		HuMoments(m, h);
		samples.at<float>(Point(0, i)) = (float)h[0];
		samples.at<float>(Point(1, i)) = (float)h[1];
		samples.at<float>(Point(2, i)) = (float)h[2];
		samples.at<float>(Point(3, i)) = (float)h[3];
		samples.at<float>(Point(4, i)) = (float)h[4];
		samples.at<float>(Point(5, i)) = (float)h[5];
		samples.at<float>(Point(6, i)) = (float)h[6];

		labels.at<float>(Point(0, i)) = 0.0f;
		labels.at<float>(Point(1, i)) = 1.0f;
		labels.at<float>(Point(2, i)) = 0.0f;
	}

	//보 학습
	Mat palm_image;
	Mat palm_handmask;
	for (; i < numSamples * 3; i++)
	{
		string sfileName = "collect/palm" + NumberToString(i + 1 - (numSamples * 2)) + ".jpg";
		palm_image = imread(sfileName,-1);
		//palm_handmask = get_CrCb_HandMask3(palm_image);
		palm_handmask = palm_image;

		Moments m = cv::moments(palm_handmask, false);
		double h[7];
		HuMoments(m, h);
		samples.at<float>(Point(0, i)) = (float)h[0];
		samples.at<float>(Point(1, i)) = (float)h[1];
		samples.at<float>(Point(2, i)) = (float)h[2];
		samples.at<float>(Point(3, i)) = (float)h[3];
		samples.at<float>(Point(4, i)) = (float)h[4];
		samples.at<float>(Point(5, i)) = (float)h[5];
		samples.at<float>(Point(6, i)) = (float)h[6];

		labels.at<float>(Point(0, i)) = 0.0f;
		labels.at<float>(Point(1, i)) = 0.0f;
		labels.at<float>(Point(2, i)) = 1.0f;
	}

	if (!nnPtr->train(samples, ml::ROW_SAMPLE, labels))	//샘플들에 대해서 분류기를 학습
		return 1;

	//랜덤 초기화
	srand((unsigned int)time(NULL));
	countdown = 0;

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void COpencv_Project_003Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 응용 프로그램의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void COpencv_Project_003Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		if (!window_image.empty())
		{
			//Mat -> IplImage
			m_pImage = &IplImage(window_image);
				
			//메인 이미지
			CDC * pDC;
			CRect rect;

			//Picture Control의 DC를 얻어옴.
			pDC = m_video.GetDC();
			//Picture Control의 사각형 영역 알아내기
			m_video.GetClientRect(&rect);
			//IplImage 타입의 이미지를 CvvImage 타입의 변수에 복사
			m_cImage.CopyOf(m_pImage);
			//CvvImage타입의 이미지를 Picture Control의 DC에 그림
			m_cImage.DrawToHDC(pDC->m_hDC, rect);

			//가져온 DC 해제
			ReleaseDC(pDC);

	
			//사용자 이미지 출력
			CDC * user_pDC;
			m_user_pImage = &IplImage(m_user_mImage);
			//m_user_mImage = imread("images/bawi_image.png");
			user_pDC = m_user_image.GetDC();
			m_user_image.GetClientRect(&rect);
			m_user_cImage.CopyOf(m_user_pImage);
			m_user_cImage.DrawToHDC(user_pDC->m_hDC, rect);
			//가져온 DC 해제
			ReleaseDC(user_pDC);

			//컴퓨터 이미지 출력
			CDC * comptuer_pDC;
			m_computer_pImage = &IplImage(m_computer_mImage);
			//m_computer_mImage = imread("images/bawi_image.png");
			comptuer_pDC = m_computer_image.GetDC();
			m_user_image.GetClientRect(&rect);
			m_computer_cImage.CopyOf(m_computer_pImage);
			m_computer_cImage.DrawToHDC(comptuer_pDC->m_hDC, rect);
			//가져온 DC 해제
			ReleaseDC(comptuer_pDC);

		}





		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR COpencv_Project_003Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void COpencv_Project_003Dlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	Mat output;
	double  minVal, maxVal;
	Point minLoc, maxLoc;
	int computer_input;
	if (nIDEvent == _DEF_WEBCAM)	//웹캠 카운터 이벤트 처리
	{
		if (cam.isOpened())
		{
			cam >> m_mImage;
			flip(m_mImage, m_mImage, 1);// flip은 영상의 좌우반전을 위해 넣은 코드이므로 필요없을 시 지워버리셔도 됩니다
			if (m_mImage.cols >= 640 || m_mImage.rows >= 480)
				resize(m_mImage, m_mImage, outputSize, 0.5, 0.5, 1);


			m_mImage.copyTo(window_image);	//화면 출력용 이미지
			Mat value(Size(inputLayerSize, 1), CV_32F);
			m_mImage = detectAndDisplay(m_mImage);		//얼굴 검출 알고리즘을 통해, 얼굴 부분을 지움.
			m_mImage = get_CrCb_HandMask3(m_mImage);	//손 검출 함수를 통해서, 손부분 검출
			double r;
			Point center;
			center = getHandCenter(m_mImage, r);		//손의 무게중심을 구한다.

			//손에 해당하는 부분만 ROI로 추출
			Mat roi;
			if (center.x - 80 >= 0 && center.y - 70 - 50 >= 0 && center.x + 200 <= 960 && center.y + 200 - 70 <= 540)
			{
				roi = m_mImage(Rect(center.x - 80, center.y - 70 - 50, 200, 200));

				//손 영역 라벨링
				Mat img_labels, stats, centroids;
				int numOfLables = connectedComponentsWithStats(m_mImage, img_labels,
					stats, centroids, 8, CV_32S);
				int left = stats.at<int>(1, CC_STAT_LEFT);
				int top = stats.at<int>(1, CC_STAT_TOP);
				int width = stats.at<int>(1, CC_STAT_WIDTH);
				int height = stats.at<int>(1, CC_STAT_HEIGHT);
				rectangle(window_image, Point(left, top), Point(left + width, top + height), Scalar(0, 255, 0), 3);


			}
			Moments m = cv::moments(roi, false);
			double h[7];
			HuMoments(m, h);
			value.at<float>(Point(0, 0)) = (float)h[0];
			value.at<float>(Point(1, 0)) = (float)h[1];
			value.at<float>(Point(2, 0)) = (float)h[2];
			value.at<float>(Point(3, 0)) = (float)h[3];
			value.at<float>(Point(4, 0)) = (float)h[4];
			value.at<float>(Point(5, 0)) = (float)h[5];
			value.at<float>(Point(6, 0)) = (float)h[6];

		
			nnPtr->predict(value, output);	//분류기를 통해서 손모양 결과를 얻음.

			minMaxLoc(output, &minVal, &maxVal, &minLoc, &maxLoc);

			//사용자 입력
			if (maxLoc.x == 0)
				m_user_mImage = imread("images/bawi_image.png");
			else if (maxLoc.x == 1)
				m_user_mImage = imread("images/gawi_image.png");
			else if (maxLoc.x == 2)
				m_user_mImage = imread("images/palm_image.png");
			
			//컴퓨터 입력
			computer_input = rand() % 3;
			if (computer_input == 0)
				m_computer_mImage = imread("images/bawi_image.png");
			else if (computer_input == 1)
				m_computer_mImage = imread("images/gawi_image.png");
			else if (computer_input == 2)
				m_computer_mImage = imread("images/palm_image.png");
			
		}
		
		//countdown이 0이되어 게임을 종료하고 게임 결과 출력
		if (countdown < 0)
		{
			KillTimer(_DEF_WEBCAM);
			KillTimer(_DEF_COUNTDOWN);

	
			if (maxLoc.x == 0 && computer_input == 0)
			{
				window_image = imread("images/draw.jpg");

			}
			else if (maxLoc.x == 0 && computer_input == 1)
			{
				window_image = imread("images/winner.jpg");
			}
			else if (maxLoc.x == 0 && computer_input == 2)
			{
				window_image = imread("images/loser.jpg");
			}
			else if (maxLoc.x == 1 && computer_input == 0)
			{
				window_image = imread("images/loser.jpg");

			}
			else if (maxLoc.x == 1 && computer_input == 1)
			{
				window_image = imread("images/draw.jpg");
			}
			else if (maxLoc.x == 1 && computer_input == 2)
			{
				window_image = imread("images/winner.jpg");
			}
			else if (maxLoc.x == 2 && computer_input == 0)
			{
				window_image = imread("images/winner.jpg");

			}
			else if (maxLoc.x == 2 && computer_input == 1)
			{
				window_image = imread("images/loser.jpg");
			}
			else if (maxLoc.x == 2 && computer_input == 2)
			{
				window_image = imread("images/draw.jpg");
			}

		}

		Invalidate(FALSE);	//화면을 다시 그린다.
	}

	if (nIDEvent == _DEF_COUNTDOWN)//카운트 다운 타이머 이벤트 처리
	{
		CString str;
		str.Format(_T("%d"), countdown);
		SetDlgItemText(IDC_COUNT_DISPLAY, str);		//카운트 다운 레이블에 카운트다운 출력
		Invalidate(FALSE);	//화면을 다시 그린다.
		countdown--;
	}


	CDialogEx::OnTimer(nIDEvent);
}


void COpencv_Project_003Dlg::OnDestroy()
{
	CDialogEx::OnDestroy();

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
	KillTimer(_DEF_WEBCAM);	//웹캠 카운터 종료
	if (cam.isOpened())
	{
		cam.release();	//웹캠 메모리 할당 해제
	}
}


BOOL COpencv_Project_003Dlg::OnEraseBkgnd(CDC* pDC)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	return CDialogEx::OnEraseBkgnd(pDC);
	//return TRUE;
}

Mat get_CrCb_HandMask3(Mat image)
{

	//컬러 공간 변환 BGR->HSV
	Mat CrCb;
	Mat HSV;
	Mat mask;
	Mat HSV_mask;

	image = detectAndDisplay(image);

	cvtColor(image, CrCb, COLOR_BGR2YCrCb);
	cvtColor(image, HSV, COLOR_BGR2HSV);

	GaussianBlur(CrCb, CrCb, Size(7, 7), 1.0); //전처리 가우시안 블러링

	Scalar lower_HSV = Scalar(0, 48, 0);
	Scalar upper_HSV = Scalar(40, 150, 255);

	Scalar lower_CrCb = Scalar(0, 120, 73);
	Scalar upper_CrCb = Scalar(255, 170, 158);
	Mat skin;

	inRange(HSV, lower_HSV, upper_HSV, HSV_mask);//지정한 색범위에 따라서, 영상을 검출하여 MASK영상
	inRange(CrCb, lower_CrCb, upper_CrCb, mask);//지정한 색범위에 따라서, 영상을 검출하여 MASK영상 생성
	bitwise_or(mask, mask, mask, HSV_mask);


	bitwise_and(image, image, skin, HSV_mask);


	cvtColor(skin, skin, COLOR_BGR2GRAY);					//피부색을 분리한 영상을 다시 그레이스케일 영상으로 변환
	//equalizeHist(skin, skin);	//히스토그램 평활화
	threshold(skin, skin, 127.0, 255.0, CV_THRESH_OTSU);		//otsu threshold(오츠 임계화)

	/*후처리(모폴로지 연산)*/
	Mat element_close = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));	//객체내부 매꾸기
	morphologyEx(skin, skin, MORPH_CLOSE, element_close);

	Mat element_open = getStructuringElement(MORPH_ELLIPSE, Size(7, 7)); //작은돌기 제거
	morphologyEx(skin, skin, MORPH_OPEN, element_open);

	Canny(skin, skin, 0, 255);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(skin, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	int s = findBiggestContour(contours);

	Mat drawing = Mat::zeros(image.size(), CV_8UC1);
	drawContours(drawing, contours, s, Scalar(255), -1, 8, hierarchy, 0, Point());

	//라벨링 한 후, 해당 넓이의 1/2 제거
	Mat img_labels, stats, centroids;
	int numOfLables = connectedComponentsWithStats(drawing, img_labels,stats, centroids, 8, CV_32S);
	int left = stats.at<int>(1, CC_STAT_LEFT);
	int top = stats.at<int>(1, CC_STAT_TOP);
	int width = stats.at<int>(1, CC_STAT_WIDTH);
	int height = stats.at<int>(1, CC_STAT_HEIGHT);

	if (width * 1.6 <= height)
		rectangle(drawing, Point(left, top + height * 0.5), Point(left + width, top + height), Scalar(0, 0, 0), -1);

	double radius;
	Point center = getHandCenter(drawing, radius);

	rectangle(drawing, Rect(0, center.y + 60, 960, 600), Scalar(0, 0, 0), -1);

	return drawing;
}


//손바닥의 중심점과 반지름 반환 함수
Point getHandCenter(const Mat& mask, double& radius)
{

	//거리 변환 행렬을 저장할 변수
	Mat dst;

	distanceTransform(mask, dst, CV_DIST_L2, 5);

	//거리 변환 행렬에서 값(거리)이 가장 큰 픽셀의 좌표와, 값을 얻어온다.

	int maxIdx[2];    //좌표 값을 얻어올 배열(행, 열 순으로 저장됨)

	minMaxIdx(dst, NULL, &radius, NULL, maxIdx, mask);   //최소값은 사용 X

	return Point(maxIdx[1], maxIdx[0]);

}

// Function detectAndDisplay(얼굴 검출 함수)
Mat detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	Mat crop;
	Mat res;
	Mat gray;
	string text;
	stringstream sstm;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(100, 100));
	if (faces.size() != 1)
		return frame;

	Point pt1(faces[0].x - 40, faces[0].y - 40); // Display detected faces on main window - live stream from camera
	Point pt2((faces[0].x + faces[0].height + 40), (faces[0].y + faces[0].width + 40));
	rectangle(frame, pt1, pt2, Scalar(0, 0, 0), -1, 8, 0);


	return frame;
}

//가장 큰 contour의 인덱스를 반환하는 함수
int findBiggestContour(vector<vector<Point> > contours){
	int indexOfBiggestContour = -1;
	int sizeOfBiggestContour = 0;

	for (int i = 0; i < contours.size(); i++){
		if (contours[i].size() > sizeOfBiggestContour){
			sizeOfBiggestContour = (int)contours[i].size();
			indexOfBiggestContour = i;
		}
	}

	return indexOfBiggestContour;
}

template <typename T>
string NumberToString(T Number)
{
	ostringstream ss;
	ss << Number;
	return ss.str();
}


void COpencv_Project_003Dlg::OnBnClickedButton()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	//캠 연결
	cam.open(0);
	if (!cam.isOpened())
	{
		//캠 연결 실패시 에러 메세지 출력
		AfxMessageBox(_T("카메라 열기 에러!"));
	}
	else
	{
		//캠 연결 성공시 타이머 설정
		AfxMessageBox(_T("게임을 시작합니다."));
		countdown = 3;
		SetTimer(_DEF_WEBCAM, 30, NULL);			//웹캠 타이머 시작
		SetTimer(_DEF_COUNTDOWN, 1000, NULL);		//카운트 다운 타이머 시작
	}


}

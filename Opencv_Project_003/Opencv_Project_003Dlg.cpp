
// Opencv_Project_003Dlg.cpp : ���� ����
//

#include "stdafx.h"
#include "Opencv_Project_003.h"
#include "Opencv_Project_003Dlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//���������� �ν��� ���� ����
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

// ���� ���α׷� ������ ���Ǵ� CAboutDlg ��ȭ �����Դϴ�.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// ��ȭ ���� �������Դϴ�.
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �����Դϴ�.

// �����Դϴ�.
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


// COpencv_Project_003Dlg ��ȭ ����



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


// COpencv_Project_003Dlg �޽��� ó����

BOOL COpencv_Project_003Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// �ý��� �޴��� "����..." �޴� �׸��� �߰��մϴ�.

	// IDM_ABOUTBOX�� �ý��� ��� ������ �־�� �մϴ�.
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

	// �� ��ȭ ������ �������� �����մϴ�.  ���� ���α׷��� �� â�� ��ȭ ���ڰ� �ƴ� ��쿡��
	//  �����ӿ�ũ�� �� �۾��� �ڵ����� �����մϴ�.
	SetIcon(m_hIcon, TRUE);			// ū �������� �����մϴ�.
	SetIcon(m_hIcon, FALSE);		// ���� �������� �����մϴ�.

	// TODO: ���⿡ �߰� �ʱ�ȭ �۾��� �߰��մϴ�.

	//���α׷� ���̺� ��Ʈ ó��
	label1.CreateFont(30, 15, 0, 0, 1000, 1, 0, 0, 0, OUT_DEFAULT_PRECIS, 0, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, _T("�������"));
	GetDlgItem(IDC_USER_LABEL)->SetFont(&label1);
	label2.CreateFont(30, 15, 0, 0, 1000, 1, 0, 0, 0, OUT_DEFAULT_PRECIS, 0, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, _T("�������"));
	GetDlgItem(IDC_COMPUTER_LABEL)->SetFont(&label2);
	label3.CreateFont(30, 15, 0, 0, 1000, 1, 0, 0, 0, OUT_DEFAULT_PRECIS, 0, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, _T("�������"));
	GetDlgItem(IDC_COUNT_LABEL)->SetFont(&label3);
	label4.CreateFont(100, 100, 0, 0, 1000, 1, 0, 0, 0, OUT_DEFAULT_PRECIS, 0, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, _T("�������"));
	GetDlgItem(IDC_COUNT_DISPLAY)->SetFont(&label4);
	label5.CreateFont(50, 25, 0, 0, 1000, 1, 0, 0, 0, OUT_DEFAULT_PRECIS, 0, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, _T("�������"));
	GetDlgItem(IDC_BUTTON)->SetFont(&label5);
	

	// Load the cascade (�� ���� �з��⸦ load �Ѵ�.)
	if (!face_cascade.load(face_cascade_name)){
		printf("--(!)Error loading\n");
		return (-1);
	}

	//�ΰ��Ű��(ANN) �з��� �ʱ�ȭ
	nnPtr->setLayerSizes(layerSizes);
	nnPtr->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);

	Mat samples(Size(inputLayerSize, numSamples * 3), CV_32F);
	//Mat image;
	Mat image;
	Mat labels(Size(outputLayerSize, numSamples * 3), CV_32F);

	//���� �н�
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

	//���� �н�
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

	//�� �н�
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

	if (!nnPtr->train(samples, ml::ROW_SAMPLE, labels))	//���õ鿡 ���ؼ� �з��⸦ �н�
		return 1;

	//���� �ʱ�ȭ
	srand((unsigned int)time(NULL));
	countdown = 0;

	return TRUE;  // ��Ŀ���� ��Ʈ�ѿ� �������� ������ TRUE�� ��ȯ�մϴ�.
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

// ��ȭ ���ڿ� �ּ�ȭ ���߸� �߰��� ��� �������� �׸�����
//  �Ʒ� �ڵ尡 �ʿ��մϴ�.  ����/�� ���� ����ϴ� MFC ���� ���α׷��� ��쿡��
//  �����ӿ�ũ���� �� �۾��� �ڵ����� �����մϴ�.

void COpencv_Project_003Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // �׸��⸦ ���� ����̽� ���ؽ�Ʈ�Դϴ�.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Ŭ���̾�Ʈ �簢������ �������� ����� ����ϴ�.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// �������� �׸��ϴ�.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		if (!window_image.empty())
		{
			//Mat -> IplImage
			m_pImage = &IplImage(window_image);
				
			//���� �̹���
			CDC * pDC;
			CRect rect;

			//Picture Control�� DC�� ����.
			pDC = m_video.GetDC();
			//Picture Control�� �簢�� ���� �˾Ƴ���
			m_video.GetClientRect(&rect);
			//IplImage Ÿ���� �̹����� CvvImage Ÿ���� ������ ����
			m_cImage.CopyOf(m_pImage);
			//CvvImageŸ���� �̹����� Picture Control�� DC�� �׸�
			m_cImage.DrawToHDC(pDC->m_hDC, rect);

			//������ DC ����
			ReleaseDC(pDC);

	
			//����� �̹��� ���
			CDC * user_pDC;
			m_user_pImage = &IplImage(m_user_mImage);
			//m_user_mImage = imread("images/bawi_image.png");
			user_pDC = m_user_image.GetDC();
			m_user_image.GetClientRect(&rect);
			m_user_cImage.CopyOf(m_user_pImage);
			m_user_cImage.DrawToHDC(user_pDC->m_hDC, rect);
			//������ DC ����
			ReleaseDC(user_pDC);

			//��ǻ�� �̹��� ���
			CDC * comptuer_pDC;
			m_computer_pImage = &IplImage(m_computer_mImage);
			//m_computer_mImage = imread("images/bawi_image.png");
			comptuer_pDC = m_computer_image.GetDC();
			m_user_image.GetClientRect(&rect);
			m_computer_cImage.CopyOf(m_computer_pImage);
			m_computer_cImage.DrawToHDC(comptuer_pDC->m_hDC, rect);
			//������ DC ����
			ReleaseDC(comptuer_pDC);

		}





		CDialogEx::OnPaint();
	}
}

// ����ڰ� �ּ�ȭ�� â�� ���� ���ȿ� Ŀ���� ǥ�õǵ��� �ý��ۿ���
//  �� �Լ��� ȣ���մϴ�.
HCURSOR COpencv_Project_003Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void COpencv_Project_003Dlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰� ��/�Ǵ� �⺻���� ȣ���մϴ�.
	Mat output;
	double  minVal, maxVal;
	Point minLoc, maxLoc;
	int computer_input;
	if (nIDEvent == _DEF_WEBCAM)	//��ķ ī���� �̺�Ʈ ó��
	{
		if (cam.isOpened())
		{
			cam >> m_mImage;
			flip(m_mImage, m_mImage, 1);// flip�� ������ �¿������ ���� ���� �ڵ��̹Ƿ� �ʿ���� �� ���������ŵ� �˴ϴ�
			if (m_mImage.cols >= 640 || m_mImage.rows >= 480)
				resize(m_mImage, m_mImage, outputSize, 0.5, 0.5, 1);


			m_mImage.copyTo(window_image);	//ȭ�� ��¿� �̹���
			Mat value(Size(inputLayerSize, 1), CV_32F);
			m_mImage = detectAndDisplay(m_mImage);		//�� ���� �˰����� ����, �� �κ��� ����.
			m_mImage = get_CrCb_HandMask3(m_mImage);	//�� ���� �Լ��� ���ؼ�, �պκ� ����
			double r;
			Point center;
			center = getHandCenter(m_mImage, r);		//���� �����߽��� ���Ѵ�.

			//�տ� �ش��ϴ� �κи� ROI�� ����
			Mat roi;
			if (center.x - 80 >= 0 && center.y - 70 - 50 >= 0 && center.x + 200 <= 960 && center.y + 200 - 70 <= 540)
			{
				roi = m_mImage(Rect(center.x - 80, center.y - 70 - 50, 200, 200));

				//�� ���� �󺧸�
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

		
			nnPtr->predict(value, output);	//�з��⸦ ���ؼ� �ո�� ����� ����.

			minMaxLoc(output, &minVal, &maxVal, &minLoc, &maxLoc);

			//����� �Է�
			if (maxLoc.x == 0)
				m_user_mImage = imread("images/bawi_image.png");
			else if (maxLoc.x == 1)
				m_user_mImage = imread("images/gawi_image.png");
			else if (maxLoc.x == 2)
				m_user_mImage = imread("images/palm_image.png");
			
			//��ǻ�� �Է�
			computer_input = rand() % 3;
			if (computer_input == 0)
				m_computer_mImage = imread("images/bawi_image.png");
			else if (computer_input == 1)
				m_computer_mImage = imread("images/gawi_image.png");
			else if (computer_input == 2)
				m_computer_mImage = imread("images/palm_image.png");
			
		}
		
		//countdown�� 0�̵Ǿ� ������ �����ϰ� ���� ��� ���
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

		Invalidate(FALSE);	//ȭ���� �ٽ� �׸���.
	}

	if (nIDEvent == _DEF_COUNTDOWN)//ī��Ʈ �ٿ� Ÿ�̸� �̺�Ʈ ó��
	{
		CString str;
		str.Format(_T("%d"), countdown);
		SetDlgItemText(IDC_COUNT_DISPLAY, str);		//ī��Ʈ �ٿ� ���̺� ī��Ʈ�ٿ� ���
		Invalidate(FALSE);	//ȭ���� �ٽ� �׸���.
		countdown--;
	}


	CDialogEx::OnTimer(nIDEvent);
}


void COpencv_Project_003Dlg::OnDestroy()
{
	CDialogEx::OnDestroy();

	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰��մϴ�.
	KillTimer(_DEF_WEBCAM);	//��ķ ī���� ����
	if (cam.isOpened())
	{
		cam.release();	//��ķ �޸� �Ҵ� ����
	}
}


BOOL COpencv_Project_003Dlg::OnEraseBkgnd(CDC* pDC)
{
	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰� ��/�Ǵ� �⺻���� ȣ���մϴ�.

	return CDialogEx::OnEraseBkgnd(pDC);
	//return TRUE;
}

Mat get_CrCb_HandMask3(Mat image)
{

	//�÷� ���� ��ȯ BGR->HSV
	Mat CrCb;
	Mat HSV;
	Mat mask;
	Mat HSV_mask;

	image = detectAndDisplay(image);

	cvtColor(image, CrCb, COLOR_BGR2YCrCb);
	cvtColor(image, HSV, COLOR_BGR2HSV);

	GaussianBlur(CrCb, CrCb, Size(7, 7), 1.0); //��ó�� ����þ� ����

	Scalar lower_HSV = Scalar(0, 48, 0);
	Scalar upper_HSV = Scalar(40, 150, 255);

	Scalar lower_CrCb = Scalar(0, 120, 73);
	Scalar upper_CrCb = Scalar(255, 170, 158);
	Mat skin;

	inRange(HSV, lower_HSV, upper_HSV, HSV_mask);//������ �������� ����, ������ �����Ͽ� MASK����
	inRange(CrCb, lower_CrCb, upper_CrCb, mask);//������ �������� ����, ������ �����Ͽ� MASK���� ����
	bitwise_or(mask, mask, mask, HSV_mask);


	bitwise_and(image, image, skin, HSV_mask);


	cvtColor(skin, skin, COLOR_BGR2GRAY);					//�Ǻλ��� �и��� ������ �ٽ� �׷��̽����� �������� ��ȯ
	//equalizeHist(skin, skin);	//������׷� ��Ȱȭ
	threshold(skin, skin, 127.0, 255.0, CV_THRESH_OTSU);		//otsu threshold(���� �Ӱ�ȭ)

	/*��ó��(�������� ����)*/
	Mat element_close = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));	//��ü���� �Ųٱ�
	morphologyEx(skin, skin, MORPH_CLOSE, element_close);

	Mat element_open = getStructuringElement(MORPH_ELLIPSE, Size(7, 7)); //�������� ����
	morphologyEx(skin, skin, MORPH_OPEN, element_open);

	Canny(skin, skin, 0, 255);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(skin, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	int s = findBiggestContour(contours);

	Mat drawing = Mat::zeros(image.size(), CV_8UC1);
	drawContours(drawing, contours, s, Scalar(255), -1, 8, hierarchy, 0, Point());

	//�󺧸� �� ��, �ش� ������ 1/2 ����
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


//�չٴ��� �߽����� ������ ��ȯ �Լ�
Point getHandCenter(const Mat& mask, double& radius)
{

	//�Ÿ� ��ȯ ����� ������ ����
	Mat dst;

	distanceTransform(mask, dst, CV_DIST_L2, 5);

	//�Ÿ� ��ȯ ��Ŀ��� ��(�Ÿ�)�� ���� ū �ȼ��� ��ǥ��, ���� ���´�.

	int maxIdx[2];    //��ǥ ���� ���� �迭(��, �� ������ �����)

	minMaxIdx(dst, NULL, &radius, NULL, maxIdx, mask);   //�ּҰ��� ��� X

	return Point(maxIdx[1], maxIdx[0]);

}

// Function detectAndDisplay(�� ���� �Լ�)
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

//���� ū contour�� �ε����� ��ȯ�ϴ� �Լ�
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
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.
	//ķ ����
	cam.open(0);
	if (!cam.isOpened())
	{
		//ķ ���� ���н� ���� �޼��� ���
		AfxMessageBox(_T("ī�޶� ���� ����!"));
	}
	else
	{
		//ķ ���� ������ Ÿ�̸� ����
		AfxMessageBox(_T("������ �����մϴ�."));
		countdown = 3;
		SetTimer(_DEF_WEBCAM, 30, NULL);			//��ķ Ÿ�̸� ����
		SetTimer(_DEF_COUNTDOWN, 1000, NULL);		//ī��Ʈ �ٿ� Ÿ�̸� ����
	}


}

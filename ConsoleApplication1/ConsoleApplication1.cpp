
#include "pch.h"
#include <iostream>
#include <core/core.hpp>
#include<highgui/highgui.hpp>
#include<opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <chrono>
#include<opencv2/calib3d.hpp>


using namespace cv;
using namespace std;

float leftIntrinsic[3][3] = { 489.47093,0,330.77481,0,490.07252,251.93051,0,0,1 };//左相机内参数矩阵
float leftDistortion[1][5] = { 0.03483 ,  0.00521  , 0.00089  , 0.00661 , 0.00000 };//左相机畸变系数
float leftTranslation[1][3] = {0,0,0};//左相机平移向量
float leftRotation[3][3] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };//左相机旋转矩阵

float rightIntrinsic[3][3] = { 500.84139,0,314.91960,0, 501.69713,246.11390,0,0,1 };//右相机内参数矩阵
float rightDistortion[1][5] = { 0.04121 ,  0.01971, - 0.00275 ,  0.00318 , 0.00000 };//右相机畸变系数
vector<float> R = { -0.01624 ,  0.01984 , 0.00070 };
Mat rightRotation;//右相机旋转矩阵
float rightTranslation[1][3] = { -63.25265  , 0.35746 , 17.45227 };//右相机平移向量

/*float leftIntrinsic[3][3] = { 1426.41869,0,918.37115,0,1425.31890,529.21718,0,0,1 };//左相机内参数矩阵
float leftDistortion[1][5] = { 0.01302, -0.01280, -0.00124, -0.00035,  0.00000 };//左相机畸变系数
float leftTranslation[1][3] = { 0,0,0 };//左相机平移向量
float leftRotation[3][3] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };//左相机旋转矩阵

float rightIntrinsic[3][3] = { 1434.10460,0,951.25524,0, 1431.61911,562.67978,0,0,1 };//右相机内参数矩阵
float rightDistortion[1][5] = { 0.01612, -0.00877  , 0.00041  , 0.00093 , 0.00000 };//右相机畸变系数
vector<float> R = { 0.01574, -0.00300, -0.00063 };
Mat rightRotation;//右相机旋转矩阵
float rightTranslation[1][3] = { 0.05682 ,-0.00356 , 0.00564 };//右相机平移向量*/

Point3f uv2xyz(Point2f uvLeft, Point2f uvRight)

{
	//  [u1]      |X|					  [u2]      |X|
	//Z*|v1| = Ml*|Y|					Z*|v2| = Mr*|Y|
	//  [ 1]      |Z|					  [ 1]      |Z|
	//			  |1|
	Mat mLeftRotation = Mat(3, 3, CV_32F, leftRotation);
	Mat mLeftTranslation = Mat(3, 1, CV_32F, leftTranslation);
	Mat mLeftRT = Mat(3, 4, CV_32F);//左相机M矩阵
	hconcat(mLeftRotation, mLeftTranslation, mLeftRT);
	Mat mLeftIntrinsic = Mat(3, 3, CV_32F, leftIntrinsic);
	Mat mLeftM = mLeftIntrinsic * mLeftRT;
	//cout<<"左相机M矩阵 = "<<endl<<mLeftM<<endl

	//Mat mRightRotation = Mat(3, 3, CV_32F, rightRotation);
	Mat mRightRotation = rightRotation;
	Mat mRightTranslation = Mat(3, 1, CV_32F, rightTranslation);
	Mat mRightRT = Mat(3, 4, CV_32F);//右相机M矩阵	
	hconcat(mRightRotation, mRightTranslation, mRightRT);
	Mat mRightIntrinsic = Mat(3, 3, CV_32F, rightIntrinsic);
	Mat mRightM = mRightIntrinsic * mRightRT;
	//cout<<"右相机M矩阵 = "<<endl<<mRightM<<endl;

	//最小二乘法A矩阵
	Mat A = Mat(4, 3, CV_32F);
	A.at<float>(0, 0) = uvLeft.x * mLeftM.at<float>(2, 0) - mLeftM.at<float>(0, 0);
	A.at<float>(0, 1) = uvLeft.x * mLeftM.at<float>(2, 1) - mLeftM.at<float>(0, 1);
	A.at<float>(0, 2) = uvLeft.x * mLeftM.at<float>(2, 2) - mLeftM.at<float>(0, 2);

	A.at<float>(1, 0) = uvLeft.y * mLeftM.at<float>(2, 0) - mLeftM.at<float>(1, 0);
	A.at<float>(1, 1) = uvLeft.y * mLeftM.at<float>(2, 1) - mLeftM.at<float>(1, 1);
	A.at<float>(1, 2) = uvLeft.y * mLeftM.at<float>(2, 2) - mLeftM.at<float>(1, 2);

	A.at<float>(2, 0) = uvRight.x * mRightM.at<float>(2, 0) - mRightM.at<float>(0, 0);
	A.at<float>(2, 1) = uvRight.x * mRightM.at<float>(2, 1) - mRightM.at<float>(0, 1);
	A.at<float>(2, 2) = uvRight.x * mRightM.at<float>(2, 2) - mRightM.at<float>(0, 2);

	A.at<float>(3, 0) = uvRight.y * mRightM.at<float>(2, 0) - mRightM.at<float>(1, 0);
	A.at<float>(3, 1) = uvRight.y * mRightM.at<float>(2, 1) - mRightM.at<float>(1, 1);
	A.at<float>(3, 2) = uvRight.y * mRightM.at<float>(2, 2) - mRightM.at<float>(1, 2);

	//最小二乘法B矩阵	
	Mat B = Mat(4, 1, CV_32F);
	B.at<float>(0, 0) = mLeftM.at<float>(0, 3) - uvLeft.x * mLeftM.at<float>(2, 3);
	B.at<float>(1, 0) = mLeftM.at<float>(1, 3) - uvLeft.y * mLeftM.at<float>(2, 3);
	B.at<float>(2, 0) = mRightM.at<float>(0, 3) - uvRight.x * mRightM.at<float>(2, 3);
	B.at<float>(3, 0) = mRightM.at<float>(1, 3) - uvRight.y * mRightM.at<float>(2, 3);

	Mat XYZ = Mat(3, 1, CV_32F);
	//采用SVD最小二乘法求解XYZ	
	solve(A, B, XYZ, DECOMP_SVD);
	//cout<<"空间坐标为 = "<<endl<<XYZ<<endl; 	
	//世界坐标系中坐标	
	Point3f world;
	world.x = XYZ.at<float>(0, 0);
	world.y = XYZ.at<float>(1, 0);
	world.z = XYZ.at<float>(2, 0);
	return world;
}

Mat jiaozheng(Mat image, float intrinsic[3][3], float distortion[1][5])
{ 
	Size image_size = image.size();   
	Mat intrinsic_matrix = Mat(3, 3, CV_32FC1, intrinsic);    
	Mat distortion_coeffs = Mat(1, 5, CV_32FC1, distortion);   
	Mat R = Mat::eye(3, 3, CV_32F);          
	Mat mapx = Mat(image_size, CV_32FC1);   
	Mat mapy = Mat(image_size, CV_32FC1);       
	initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);   
	Mat t = image.clone();   
	cv::remap(image, t, mapx, mapy, INTER_LINEAR);   
	return t; 
}

int main()
{
	Rodrigues(R, rightRotation);//旋转向量到旋转矩阵
	//读取标记
	Mat mark = imread("chessboard.bmp", IMREAD_GRAYSCALE);
	threshold(mark, mark, 100, 255, THRESH_BINARY_INV);
	vector<vector<Point>> markc;
	findContours(mark,markc , RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
	/*Moments mom = moments(markc, false);
	double humom[7];
	HuMoments(mom, humom);*/
	drawContours(mark, markc, 0, Scalar(0, 0, 255), 1, 8);
	imshow("chessboard", mark);
	//打开第一个摄像头	
	VideoCapture cap1(1);
	VideoCapture cap2(2);
	//判断摄像头是否打开	
	if (!cap1.isOpened())
	{
		cout << "摄像头1未成功打开" << endl;
		if (!cap2.isOpened())
		{
			cout << "摄像头2未成功打开" << endl;
		}
	}

	for (;;)
	{
		//auto start = std::chrono::steady_clock::now();//计时器
		//创建Mat对象		
		Mat frame1;			//从cap中读取一帧存到frame中		
		cap1 >> frame1;		
		Mat frame2;
		cap2 >> frame2;
		int h = frame1.size().height;
		if (frame1.empty())//判断是否读取到
		{
			break;
		}
		if (frame2.empty())
		{
			break;
		}
		jiaozheng(frame1, rightIntrinsic, rightDistortion);//校正
		jiaozheng(frame2, leftIntrinsic, leftDistortion);

		//medianBlur(frame1,frame1,5);
		Mat kernel(3, 3, CV_32F, Scalar(-1));		//拉普拉斯锐化
		kernel.at<float>(1, 1) = 8.9;
		filter2D(frame1, frame1, frame1.depth(), kernel);
		//GaussianBlur(frame2, frame2, Size(3, 3), 0, 0);
		//medianBlur(frame2, frame2, 5);
		filter2D(frame2, frame2, frame2.depth(), kernel);


		Mat gray1, gray2;
		cvtColor(frame1, gray1, COLOR_BGR2GRAY);
		cvtColor(frame2, gray2, COLOR_BGR2GRAY);
		Mat twin1, twin2;
		threshold(gray1, twin1, 100, 255, THRESH_BINARY_INV);//二值化
		threshold(gray2, twin2, 100, 255, THRESH_BINARY_INV);
		//adaptiveThreshold(gray1, twin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 60);
		blur(twin1, twin1, Size(3, 3));
		blur(twin2, twin2, Size(3, 3));
		/*//腐蚀
		Mat kernelerode = getStructuringElement(MORPH_RECT, Size(5, 5));
		erode(twin,twin, kernelerode);*/
		/*//霍夫变换找圆心√
		vector<Vec3f> circles1, circles2;
		HoughCircles(twin1 , circles1, HOUGH_GRADIENT, 1, h / 64, 150, 70, h / 64, h / 8);
		//vector<Vec3f>::const_iterator it1 = circles1.begin();
		for (size_t i = 0; i < circles1.size(); i++)
		{
			Point center(round(circles1[i][0]), round(circles1[i][1]));
			int radius = round(circles1[i][2]);
			circle(frame1, center, 3, Scalar(0, 255, 0), -1, 4, 0);
			circle(frame1, center, radius, Scalar(0, 0, 255), 3, 4, 0);
		}
		vector<Vec3f> ;
		HoughCircles(twin2 , circles2, HOUGH_GRADIENT, 1, h / 64, 150, 70, h / 64, h / 8);
		//vector<Vec3f>::const_iterator it2 = circles2.begin();
		for (size_t i = 0; i < circles2.size(); i++)
		{
			Point center(round(circles2[i][0]), round(circles2[i][1]));
			int radius = round(circles2[i][2]);
			circle(frame2, center, 3, Scalar(0, 255, 0), -1, 4, 0);
			circle(frame2, center, radius, Scalar(0, 0, 255), 3, 4, 0);
		}
		*/


		//2*2棋盘搜索√
		vector<vector<Point>> corners1, corners2;
		//vector<Vec4i> hierarcy;
		findContours(twin1, corners1, RETR_CCOMP, CHAIN_APPROX_SIMPLE);//提取轮廓
		findContours(twin2, corners2, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
				
		//std::vector<cv::Point> roi_point_approx;
		//cv::Mat roi_approx(twin.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		//vector<std::vector<cv::Point>> roi_point_approx(corners.size());
		std::vector<std::vector<cv::Point>>::iterator itc = corners1.begin();
		for (; itc != corners1.end();)//轮廓筛选
		{
			double area = contourArea(*itc);
			if (area < 400)//面积
			{
				itc = corners1.erase(itc);
			}
			else
			{
				double dist = matchShapes(*itc, markc[0], CONTOURS_MATCH_I2, 0);//I1,I2都行，3不行
				//printf("contour match distance : %.2f\n", dist);
				if (dist > 0.16)//匹配度
				{
					itc = corners1.erase(itc);
				}
				else
				{
					++itc;
				}
			}
		}
		std::vector<std::vector<cv::Point>>::iterator itc2 = corners2.begin();
		for (; itc2 != corners2.end();)//轮廓筛选
		{
			double area = contourArea(*itc2);
			if (area < 400)//面积
			{
				itc2 = corners2.erase(itc2);
			}
			else
			{
				double dist = matchShapes(*itc2, markc[0], CONTOURS_MATCH_I2, 0);
				//printf("contour match distance : %.2f\n", dist);
				if (dist > 0.16)//匹配度
				{
					itc2 = corners2.erase(itc2);
				}
				else
				{
					++itc2;
				}
			}
		}
		drawContours(frame2, corners2, -1, Scalar(0, 255, 255), 2, 8);  //轮廓绘制
		//approxPolyDP(Mat(corners[i]), roi_point_approx[i], 7, true);
		drawContours(frame1, corners1, -1, Scalar(0, 255, 255), 2, 8);  //轮廓绘制
		//中心点提取

		vector<Point> TP1(corners1.size());
		for (int i = 0; i < corners1.size(); i++)//寻找轮廓中心点
		{
			Moments moment = moments(corners1[i], false);
			if (moment.m00 != 0)//除数不能为0
			{
				TP1[i].x = cvRound(moment.m10 / moment.m00);//计算重心横坐标
				TP1[i].y = cvRound(moment.m01 / moment.m00);//计算重心纵坐标
			}
			cv::circle(frame1, TP1[i], 2, cv::Scalar(100 * i, 50 * i, 255));
		}
		//vector<Point> RTP1(TP1.size());
		vector<Point2f> TP2(corners2.size());
		for (int i = 0; i < corners2.size(); i++)//寻找轮廓中心点
		{
			Moments moment = moments(corners2[i], false);
			if (moment.m00 != 0)//除数不能为0
			{
				TP2[i].x = cvRound(moment.m10 / moment.m00);//计算重心横坐标
				TP2[i].y = cvRound(moment.m01 / moment.m00);//计算重心纵坐标
			}
			cv::circle(frame2, TP2[i], 2, cv::Scalar(100 * i, 50 * i, 255));
		}

		if (TP1.size() > 1)
		{
			for (int i = 1; i < TP1.size(); i++)//重排匹配
			{
				if (TP1[i].x < TP1[i - 1].x)
				{
					int j = i;
					for (; j - 1 >= 0 && TP1[j - 1].x > TP1[j].x; j--)
					{
						Point T = TP1[j];
						TP1[j] = TP1[j - 1];
						TP1[j - 1] = T;
					}
				}
			}
		}
		//左
		if (TP2.size() > 1)
		{
			for (int i = 1; i < TP2.size(); i++)//重排匹配
			{
				if (TP2[i].x < TP2[i - 1].x)
				{
					int j = i;
					for (; j - 1 >= 0 && TP2[j - 1].x > TP2[j].x; j--)
					{
						Point T = TP2[j];
						TP2[j] = TP2[j - 1];
						TP2[j - 1] = T;
					}
				}
			}
		}

		/*//template模板匹配
		cv::Mat image_matched;
		matchTemplate(gray1, mark, image_matched, TM_CCOEFF_NORMED);//无旋转不变性
		cv::imshow("match result", image_matched);
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;	//寻找最佳匹配位置
		cv::minMaxLoc(image_matched, &minVal, &maxVal, &minLoc, &maxLoc);
		rectangle(frame1,cv::Point(maxLoc.x + mark.cols/2, maxLoc.y + mark.rows/2), cv::Point(maxLoc.x - mark.cols / 2, maxLoc.y - mark.rows / 2), cv::Scalar(0, 0, 255), 2,  8,  0);
		*/

		//四边形角点识别
		/*auto i = corners.begin();
		approxPolyDP(*i, roi_point_approx, 7, 1);
		for (auto a : roi_point_approx)
			cv::circle(frame1, a, 2, cv::Scalar(0, 0, 255));
		//cv::imshow("roi_approx", roi_approx);*/


		//对应点三维坐标计算
		if (2==TP2.size()&& 2 == TP1.size())
		{
			
			Point3f worldPoint1,worldPoint2;
			worldPoint1 = uv2xyz(TP2[0], TP1[0]);
			worldPoint2 = uv2xyz(TP2[1], TP1[1]);
			cout << worldPoint1 <<worldPoint2<< endl;	
		}

		//imshow("gray1", gray1);//显示图片
		imshow("right", frame1);
		imshow("left", frame2);
		//imshow("twin", twin1);
		//等待50毫秒，如果按键则退出循环		
		//计时器
		/*auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double, std::micro> elapsed = end - start; // std::micro 表示以微秒为时间单位
		std::cout << "time: " << elapsed.count() << "us" << std::endl;*/

		if (waitKey(50) >= 0)
		{
			break;
		}
	}
	//system("pause");
	return 0;
}


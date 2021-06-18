#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
//#include "mask_generator.h"
#include "mask_generator.h"
#include <iostream>
//#include <opencv2\imgproc\imgproc.hpp>
using namespace cv;
#ifndef __GUI_H__
#define __GUI_H__

static Mat main_img;
static vector<Point2f> contour;

static void onMouse(int Event,int x,int y,int flags,void* param )
{

	switch(Event){
		case CV_EVENT_MOUSEMOVE:
			if(contour.size()>1){
				Mat tmp_img = main_img.clone();
				line(tmp_img, contour[contour.size()-1], 
					Point2f(x, y), Scalar(0,255,0), 1);
				imshow("project", tmp_img);
			}
			break;
		case CV_EVENT_LBUTTONDOWN:
			contour.push_back(Point2f(x, y));
			circle(main_img, Point2f(x, y), 2, Scalar(255, 0, 0));
			if(contour.size()>1){
				line(main_img, contour[contour.size()-1], 
					contour[contour.size()-2], Scalar(0,255,0), 1);
				imshow("project", main_img);
			}
			break;
		case CV_EVENT_RBUTTONDOWN:
			line(main_img, contour[contour.size()-1], 
					contour[0], Scalar(0,255,0), 1);

	}
}


typedef enum {
	COLOR, DEPTH, OUTMASK
}ImgType;


class GUI{
public:
	void run(Mat **imgs, Mat **imgs_depth, Mat **&out_imgs, Mat **&out_imgs_depth, Mat **&out_imgs_mask, int u, int v){
		namedWindow("project", 1);
		cvSetMouseCallback("project",onMouse,0);
		// light field viewer
		//copyLightField(imgs, out_imgs, imgs_depth, out_imgs_depth, u, v);
		randomLightField(out_imgs, out_imgs_depth, out_imgs_mask, u, v, imgs[0][0].rows, imgs[0][0].cols);
		int x = v/2, y = u/2;
		ImgType show = COLOR;
		int showcount = 0;
		dispalyImg(imgs[x][y], out_imgs[x][y], main_img, show);
		//main_img = imgs[x][y].clone();
		bool loop_exit = false;
		Mat *obj;
		Mat object_mask;
		Point2f shift;
		Point2f position;
		while(1){
			
			imshow( "project", main_img );
			int k = cvWaitKey(0);
			switch(k){
				case 'c':
					copyLightField(imgs, out_imgs, imgs_depth, out_imgs_depth, u, v);
					break;
				case 'd':
					y = y<u-1 ? y+1 : y;
					break;
				case 'a':
					y = y>0 ? y-1 : y;
					break;
				case 'w':
					x = x<v-1 ? x+1 : x;
					break;
				case 's':
					x = x>0 ? x-1 : x;
					break;
				case 'p':
					showcount++;
					break;
				case 'q':
					loop_exit = true;
					break;
				case 't':
					cvSetMouseCallback("project", NULL,0);
					obj = object_mask_gen(imgs[x][y], contour, position, COLOR);
					mask::find_mask_by_contour(contour, obj->cols, obj->rows, object_mask);
					shift = run_paste(imgs[x][y], out_imgs[x][y], *obj, object_mask, position);
					std::cout << "run paste done" << std::endl;
					transferAndPaste(imgs, imgs_depth, out_imgs, out_imgs_depth, out_imgs_mask, u, v, 
									*obj, contour, shift);
					cvSetMouseCallback("project",onMouse,0);
					contour.clear();
					break;
				case 'r':
					obj = object_mask_gen(imgs[x][y], contour, position, COLOR);
					//mask::find_mask_by_contour(contour, obj->cols, obj->rows, object_mask);
					deleteRegion(imgs, imgs_depth, out_imgs, out_imgs_depth, u, v, *obj, contour);
					cvSetMouseCallback("project",onMouse,0);
					contour.clear();
					break;
			}
			//main_img = imgs[x][y].clone();
			switch(showcount%3){
			case 0:
				show = COLOR;
				dispalyImg(imgs[x][y], out_imgs[x][y], main_img, show);
				break;
			case 1:
				show = DEPTH;
				dispalyImg(imgs_depth[x][y], out_imgs_depth[x][y], main_img, show);
				break;
			case 2:
				show = OUTMASK;
				dispalyImg(imgs_depth[x][y], out_imgs_mask[x][y], main_img, show);
				break;
			}
			if(loop_exit)
				break;
		}
	}
private:
	void copyLightField(Mat **imgs, Mat **&out_imgs, Mat **imgs_depth, Mat **&out_imgs_depth, int u, int v);
	void randomLightField(Mat **&out_imgs, Mat **&out_imgs_depth, Mat **&out_imgs_mask, int u, int v, int row, int col);
	void dispalyImg(Mat &imgs, Mat &out_imgs, Mat &main_img, ImgType type);
	Mat *object_mask_gen(Mat &img, vector<Point2f> &contour, Point2f &center, ImgType type);
	void pasteImg(Mat &main_img, Mat &obj, Mat &object_mask, Point2f &obj_position);
	Point2f run_paste(Mat &img, Mat &out_img, Mat &obj, Mat &object_mask, Point2f &obj_position);
	void transferAndPaste(Mat **src, Mat **src_depth, Mat **tar, Mat **tar_depth, Mat **tar_mask, int u, int v, 
					  Mat &object, vector<Point2f> &object_contour, Point2f shift);
	void deleteRegion(Mat **src, Mat **src_depth, Mat **tar, Mat **tar_depth, int u, int v, 
					  Mat &object, vector<Point2f> &object_contour);
	//void onMouse(int Event,int x,int y,int flags,void* param);
};
#endif
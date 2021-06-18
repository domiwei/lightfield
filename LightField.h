#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "4DPoint.h"
#include "LightField2.h"
//#include <iostream>
#ifndef __LIGHTFIELD_H__
#define __LIGHTFIELD_H__

using namespace cv;
typedef enum MASK_FLAG{
	MASK, NON_MASK
}MaskFlag;

inline bool check_bound(int px, int py, int x, int y)
{
	return px<x && px>=0 && py<y &&py>=0;
}

struct LightField{
	LightField(){}
	LightField(Mat **img, Mat **mask, int u, int v, Mat **depth = NULL)
	{
		cvimg = img;
		cvmask = mask;
		cvdepth = depth;
		//std::cout <<"constructor1" <<std::endl;
		if(depth!=NULL)
			this->depth = new LightFieldData<uchar, 1>(depth, u, v);
		//	std::cout <<"constructor2" <<std::endl;
		this->img = new LightFieldData<uchar, 3>(img, u, v);
		if(mask)
			this->mask = new LightFieldData<uchar, 1>(mask, u, v);
		this->u = u;
		this->v = v;
		row = img[0][0].rows;
		col = img[0][0].cols;
	}
	Scalar get_value(Point4D p, MaskFlag flag = MASK)
	{
		if(flag == NON_MASK)
			return (img->get_value(p));
		if(mask->get_value(p)[0] <100){
			return (img->get_value(p));
		}else{
			return Scalar(-1,-1,-1);
		}	
	}
	uchar getDepth(Point4D p, MaskFlag flag = MASK)
	{
		if(flag == NON_MASK)
			return (depth->get_value(p)[0]);
		if(mask->get_value(p)[0] <100){
			return (depth->get_value(p)[0]);
		}else{
			return -1;
		}
	}
	bool in_mask(Point4D &p)
	{
		if(mask->get_value(p)[0]>100)
			return true;
		else
			return false;
	}
	void set_value(Point4D p, Scalar color)
	{
		img->set_value(p, color);
		//for(int ch=0 ; ch<3 ; ch++)
			//img[p.u][p.v].at<Vec3b>(p.y, p.x)[ch] = color[ch];
	}
	void setDepth(Point4D p, uchar d)
	{
		depth->set_value(p, Scalar(d));
	}
	bool check_4D_bound(Point4D p)
	{
		return (check_bound(p.u, p.v, u, v) && check_bound(p.y, p.x, row, col));
	}

	void writeImage(char filename[30], int du, int dv)
	{
		Mat outimg(row, col, CV_8UC3);
		Mat depthimg(row, col, CV_8UC1);
		for(int y=0 ; y<row ; y++){
			for(int x=0 ; x<col ; x++){
				Scalar color = img->get_value(Point4D(du, dv, y, x));
				outimg.at<Vec3b>(y, x)[0] = color[0];
				outimg.at<Vec3b>(y, x)[1] = color[1];
				outimg.at<Vec3b>(y, x)[2] = color[2];
				depthimg.at<uchar>(y, x) = depth->get_value(Point4D(du, dv, y, x))[0];
			}
		}
		std::cout << "........." <<std::endl;
		string depth_filename = string(filename) + ".bmp";
		//imwrite(depth_filename.c_str(), depthimg);
		imwrite(filename, outimg);
	}
	LightFieldData<uchar, 3> *img;
	LightFieldData<uchar, 1> *mask;
	LightFieldData<uchar, 1> *depth;
	Mat **cvimg;
	Mat **cvmask;
	Mat **cvdepth;
	int row, col;
	int u, v;
};

#endif
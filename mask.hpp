#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
/********* for debugging ***********/
#ifndef __MASK__
#define __MASK__
namespace mask{
inline bool check_point(Point2f p, int x, int y)
{
	return p.x<x && p.x>=0 && p.y<y &&p.y>=0;
}

inline bool check_point(int px, int py, int x, int y)
{
	return px<x && px>=0 && py<y &&py>=0;
}

/* draw mask */

static void draw_line(vector<Point2f> point, Mat &mask)
{
	for(int i=0 ; i<point.size()-1 ; i++)
		line(mask, point[i], point[i+1], Scalar(255, 0, 0));
	line(mask, point[point.size()-1], point[0], Scalar(255, 0, 0));
}

static void find_mask_by_contour(vector<Point2f> point, int col, int row, Mat &mask)
{
	mask = Mat::zeros(row, col, CV_8UC1);
	draw_line(point, mask);
	vector<Point2f> flip;
	flip.push_back(Point2f(0, 0));
	while(flip.size()){
		Point2f p = flip.back();
		flip.pop_back();
		mask.at<uchar>(p.y, p.x) = 255;
		if(check_point(p.x+1, p.y, col, row) && mask.at<uchar>(p.y, p.x+1)==0)  //black
			flip.push_back(Point2f(p.x+1, p.y));
		if(check_point(p.x-1, p.y, col, row) && mask.at<uchar>(p.y, p.x-1)==0)  //black
			flip.push_back(Point2f(p.x-1, p.y));
		if(check_point(p.x, p.y+1, col, row) && mask.at<uchar>(p.y+1, p.x)==0)  //black
			flip.push_back(Point2f(p.x, p.y+1));
		if(check_point(p.x, p.y-1, col, row) && mask.at<uchar>(p.y-1, p.x)==0)  //black
			flip.push_back(Point2f(p.x, p.y-1));
	}
	// flip again
	for(int j=0 ; j<row ; j++){
		for(int i=0 ; i<col ; i++){
			if(mask.at<uchar>(j, i)==255)
				mask.at<uchar>(j, i)=0;
			else
				mask.at<uchar>(j, i)=255;
		}
	}
	// draw contour again
	draw_line(point, mask);
}
}

#endif
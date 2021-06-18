//#include <stdio.h>
#include <iostream>
//#include "mask_generator.h"
//#include "4DPatchMatch.h"
#include "DepthEstimator.h"
//#include "LightField.h"
#include "LightFieldRetarget.h"
#include "LightFieldReshuffle.h"
#include "GUI.h"
using namespace cv; 
vector<Point2f> position;
//Mat main_img;
//bool loop_exit = false;
void onMouseInpainting(int Event,int x,int y,int flags,void* param );
Mat *object_mask_gen(Mat &img);
void combine_to_big(Mat **imgs, Mat **imgs_mask, Mat &big_img, Mat &big_mask, int row, int col);
#ifndef DEBUG
#define DEBUG
#endif
/** @function main */
#define ROW 4
#define COLUMN 4
//#define WRITE_MASK
int main( int argc, char ** argv )
{
	float k = 1;
	int row=ROW, column=COLUMN;
	Mat **imgs = new Mat*[row];
	Mat **imgs_mask = new Mat*[row];
	Mat **imgs_depth = new Mat*[row];
	for(int i=0 ; i<row ; i++){
		imgs[i] = new Mat[column];
		imgs_mask[i] = new Mat[column];
		imgs_depth[i] = new Mat[column];
	}
	for(int i=0 ; i<row ; i++)
		for(int j=0 ; j<column ; j++){
			char filename[20];
			sprintf(filename, "test_hci/%d_%d.jpg", i, j); //test/%d_%d.jpg
			std::cout << i << j << std::endl;
			imgs[i][j] = imread(filename, CV_LOAD_IMAGE_COLOR );
			//resize(imgs[i][j], imgs[i][j], Size(imgs[i][j].cols*k, imgs[i][j].rows*k));
		}
	
#ifndef DEBUG
	//Mat object = imread("test/0_0_mymskimg.jpg", CV_LOAD_IMAGE_COLOR);
	//Mat object_mask = imread("test/0_0_mymsk.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	namedWindow("test", 1);
	cvSetMouseCallback("test",onMouseInpainting,0);
	//cvSetMouseCallback("test2",onMouse,0);
	// light field viewer
	int x = COLUMN/2, y = ROW/2;
	bool loop_exit = false;
	main_img = imgs[x][y].clone();
	//main_img2 = imgs[x][y].clone();
	while(1){
		imshow( "test", main_img );
		//imshow( "test", main_img2 );
		int k = cvWaitKey(0);
		switch(k){
			case 'd':
				y = y<row-1 ? y+1 : y;
				break;
			case 'a':
				y = y>0 ? y-1 : y;
				break;
			case 'w':
				x = x<column-1 ? x+1 : x;
				break;
			case 's':
				x = x>0 ? x-1 : x;
				break;
			case 'q':
				loop_exit = true;
				break;
		}
		main_img = imgs[x][y].clone();
		//main_img = imgs[x][y].clone();
		if(loop_exit)
			break;
		
	}
	Mat *obj = object_mask_gen(imgs[x][y]);
	imshow( "test", *obj);
	cvWaitKey(0);
	generate_lightfield_mask(imgs, imgs_mask, row, column, *obj, position);
#ifdef WRITE_MASK
	for(int i=0 ; i<row ; i++)
		for(int j=0 ; j<column ; j++){
			char filename[20];
			sprintf(filename, "mask/%d_%d_mask.bmp", i, j);
			std::cout << i << j << std::endl;
			imwrite(filename, imgs_mask[i][j]);
		}
#endif
	std::cout << "generate masks done" << std::endl;
#endif
#ifdef DEBUG
	/* Read the masks */
	for(int i=0 ; i<row ; i++)
		for(int j=0 ; j<column ; j++){
			char filename[20];
			sprintf(filename, "test_hci/%d_%d_mask.bmp", i, j); //"test_bud/%d_%d_mask.jpg"
			std::cout << i << j << std::endl;   
			imgs_mask[i][j] = imread(filename, CV_LOAD_IMAGE_UNCHANGED );
			//resize(imgs_mask[i][j], imgs_mask[i][j], Size(imgs_mask[i][j].cols*k, imgs_mask[i][j].rows*k));
			sprintf(filename, "depth_%d_%d.bmp", i, j); //"bud_depth/depth_%d_%d.bmp"
			imgs_depth[i][j] = imread(filename, CV_LOAD_IMAGE_UNCHANGED );
			//resize(imgs_depth[i][j], imgs_depth[i][j], Size(imgs_depth[i][j].cols*k, imgs_depth[i][j].rows*k));
		}
#endif
	
	//LightFieldData<uchar, 3> lf2(imgs, row, column);
	//DepthEstimator de(lf2.size);
//	de.estimate(lf2);
	
	for(int i=0 ; i<row ; i++)
		for(int j=0 ; j<column ; j++){
	//		imgs_depth[i][j] = de.showDepthMap(i, j);
			resize(imgs[i][j], imgs[i][j], Size(imgs[i][j].cols*k, imgs[i][j].rows*k));
			resize(imgs_depth[i][j], imgs_depth[i][j], Size(imgs_depth[i][j].cols*k, imgs_depth[i][j].rows*k));
			resize(imgs_mask[i][j], imgs_mask[i][j], Size(imgs_mask[i][j].cols*k, imgs_mask[i][j].rows*k));
		}
		//system("pause");

	/*	
	Mat **init_imgs;
	Mat **init_imgs_depth;
	Mat **init_imgs_mask;
	GUI gui;
	gui.run(imgs, imgs_depth, init_imgs, init_imgs_depth, init_imgs_mask, row, column);
	LightField lf = LightField(imgs, init_imgs_mask, row, column, imgs_depth);
	LightField tar_lf = LightField(init_imgs, init_imgs_mask, row, column, init_imgs_depth);
	LightFieldReshuffle lfshuffle;
	lfshuffle.reshuffling(lf, tar_lf, Point4D(1, 1, 3, 3), 6);
	*/


	//std::cout << lf.u << ", " << lf.v << ", " << lf.col << ", " << lf.row << std::endl;
	//LightField lf = LightField(imgs, imgs_mask, row, column, imgs_depth);
	//LightFieldRetarget lfr;
	//LightField *result_lf = lfr.retargeting(lf, 1.0, 0.75, Point4D(1,1,3,3), 6);
	//system("pause");
	
	LightField lf = LightField(imgs, imgs_mask, row, column, imgs_depth);
	PatchMatch pm;
	clock_t start_time, end_time;
	start_time = clock();
	pm.run_4d_patchmatch(lf, 1, 1, 3, 3, 5);
	end_time = clock();
	std::cout << "total time = " << (float)(end_time - start_time)/CLOCKS_PER_SEC << std::endl;
	
	//imshow("test", lf.img[0][0]);
	//waitKey(0);
	
	for(int i=0 ; i<row ; i++)
		for(int j=0 ; j<column ; j++){
			char out_name[30];
			sprintf(out_name, "out6/%d_%d.jpg", i, j);
			lf.writeImage(&out_name[0], i, j);
		}

	/* combine to big image */
//	Mat big_mask = Mat::zeros(imgs[0][0].rows*row, imgs[0][0].cols*column, CV_8UC1);
//	Mat big_img = Mat(imgs[0][0].rows*row, imgs[0][0].cols*column, CV_8UC3);
//	combine_to_big(imgs, imgs_mask, big_img, big_mask, row, column);
	
#ifndef DEBUG
/*	for(int i=0 ; i<row ; i++)
		for(int j=0 ; j<column ; j++){
			char filename[20];
			sprintf(filename, "test/%d_%d_mask.jpg", i, j);
			std::cout << i << j << std::endl;
			imwrite(filename, imgs_mask[i][j]);
		}*/
#endif
	/* Run PatchMatch */
	
 }


 
static void onMouseInpainting(int Event,int x,int y,int flags,void* param )
{

	switch(Event){
		case CV_EVENT_MOUSEMOVE:
			if(position.size()>1){
				Mat tmp_img = main_img.clone();
				line(tmp_img, position[position.size()-1], 
					Point2f(x, y), Scalar(0,255,0), 1);
				imshow("test", tmp_img);
			}
			break;
		case CV_EVENT_LBUTTONDOWN:
			position.push_back(Point2f(x, y));
			circle(main_img, Point2f(x, y), 2, Scalar(255, 0, 0));
			if(position.size()>1){
				line(main_img, position[position.size()-1], 
					position[position.size()-2], Scalar(0,255,0), 1);
				imshow("test", main_img);
			}
			break;
		case CV_EVENT_RBUTTONDOWN:
			line(main_img, position[position.size()-1], 
					position[0], Scalar(0,255,0), 1);

	}
}

Mat *object_mask_gen(Mat &img)
{
	int max_x=-1, min_x=2000, max_y=-1, min_y=2000;
	for(int i=0 ; i<position.size() ; i++){
		if(max_x < position[i].x)
			max_x = position[i].x;
		if(max_y < position[i].y)
			max_y = position[i].y;
		if(min_x > position[i].x)
			min_x = position[i].x;
		if(min_y > position[i].y)
			min_y = position[i].y;
	}
	printf("%d, %d, %d, %d\n", max_x, min_x, max_y, min_y);
	int size_buffer = 20;
	Mat *object = new Mat(max_y-min_y+size_buffer, max_x-min_x+size_buffer, CV_8UC3);
	for(int i=0 ; i<max_x-min_x+size_buffer ; i++){
		for(int j=0 ; j<max_y-min_y+size_buffer ; j++){
			//printf("%d, %d---%d, %d, %d, %d\n", i, j, max_x, min_x, max_y, min_y);
				object->at<cv::Vec3b>(j, i)=  img.at<cv::Vec3b>(min_y + j - size_buffer/2, min_x + i - size_buffer/2);
		}
	}
	for(int i=0 ; i<position.size() ; i++){
		position[i].x -= min_x-size_buffer/2;
		position[i].y -= min_y-size_buffer/2;
	}
	return object;
}

void combine_to_big(Mat **imgs, Mat **imgs_mask, Mat &big_img, Mat &big_mask, int row, int col)
{
	int img_row = imgs_mask[0][0].rows;
	int img_col = imgs_mask[0][0].cols;
	for(int u=0 ; u<row ; u++)
		for(int v=0 ; v<col ; v++){
			std::cout << "(u, v) = " << u << v << std::endl;
			for(int i=0 ; i<img_row ; i++){
				for(int j=0 ; j<img_col ; j++){
					//std::cout << "(i, j) = " << i << ", " << j << std::endl;
					big_mask.at<uchar>(img_row*u + i, img_col*v + j)= imgs_mask[u][v].at<uchar>(i, j);
					for(int ch=0 ; ch<3 ; ch++)
						big_img.at<Vec3b>(img_row*u + i, img_col*v + j)[ch] = imgs[u][v].at<Vec3b>(i, j)[ch];
						//.at(row_index, col_index)
					
				}
			}
		}
	imwrite("big_img.jpg", big_img);
	imwrite("big_mask.jpg", big_mask);
}
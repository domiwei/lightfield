#include "GUI.h"

void GUI::pasteImg(Mat &main_img, Mat &obj, Mat &object_mask, Point2f &obj_position)
	{
		for(int i=0 ; i<obj.rows ; i++)
			for(int j=0 ; j<obj.cols ; j++)
				if(object_mask.at<uchar>(i, j)>100)
					main_img.at<Vec3b>(i+obj_position.y, j+obj_position.x) = obj.at<Vec3b>(i, j);
	}

void paste2Target(Mat &tar, Mat &tar_depth, Mat &tar_mask, Mat &obj, Mat &obj_depth, Mat &obj_mask, Point2f &position)
{
	for(int i=0 ; i<obj.rows ; i++)
		for(int j=0 ; j<obj.cols ; j++)
			if(obj_mask.at<uchar>(i, j)>100){
				tar.at<Vec3b>(i+position.y, j+position.x) = obj.at<Vec3b>(i, j);
				tar_depth.at<uchar>(i+position.y, j+position.x) = obj_depth.at<uchar>(i, j);
				tar_mask.at<uchar>(i+position.y, j+position.x) = 255;
			}
}

Point2f GUI::run_paste(Mat &img, Mat &out_img, Mat &obj, Mat &object_mask, Point2f &obj_position)
{
		
		Point2f shift(0, 0);
		shift += Point2f(img.cols+30, 0);
		dispalyImg(img, out_img, main_img, COLOR);
		pasteImg(main_img, obj, object_mask, shift+obj_position);
		bool loop_exit = false;
		while(1){
			imshow( "project", main_img );
			int k = cvWaitKey(0);
			switch(k){
				case 'd':
					if(shift.x+obj_position.x < main_img.cols-1)
						shift += Point2f(1, 0);
					break;
				case 'a':
					if(shift.x+obj_position.x > 0)
						shift += Point2f(-1, 0);
					break;
				case 'w':
					if(shift.y+obj_position.y > 0)
						shift += Point2f(0, -1);
					break;
				case 's':
					if(shift.y+obj_position.y < main_img.rows-1)
						shift += Point2f(0, 1);
					break;
				case 'q':
					loop_exit = true;
					break;
			}
			dispalyImg(img, out_img, main_img, COLOR);
			pasteImg(main_img, obj, object_mask, shift+obj_position);
			if(loop_exit)
				break;
	}
	//cvDestroyWindow("project");
	return shift-Point2f(img.cols+30, 0);
}

Mat *GUI::object_mask_gen(Mat &img, vector<Point2f> &contour, Point2f &center, ImgType type)
{
	int max_x=-1, min_x=2000, max_y=-1, min_y=2000;
	for(int i=0 ; i<contour.size() ; i++){
		if(max_x < contour[i].x)
			max_x = contour[i].x;
		if(max_y < contour[i].y)
			max_y = contour[i].y;
		if(min_x > contour[i].x)
			min_x = contour[i].x;
		if(min_y > contour[i].y)
			min_y = contour[i].y;
	}
	int size_buffer = 20;
	printf("%d, %d, %d, %d\n", max_x, min_x, max_y, min_y);
	center = Point2f(min_x-size_buffer/2, min_y-size_buffer/2);
	Mat *object = new Mat(max_y-min_y+size_buffer, max_x-min_x+size_buffer, CV_8UC3);
	for(int i=0 ; i<max_x-min_x+size_buffer ; i++){
		for(int j=0 ; j<max_y-min_y+size_buffer ; j++){
			//printf("%d, %d---%d, %d, %d, %d\n", i, j, max_x, min_x, max_y, min_y);
			if(type==COLOR)
				object->at<cv::Vec3b>(j, i)=  img.at<cv::Vec3b>(min_y + j - size_buffer/2, min_x + i - size_buffer/2);
			else
				object->at<uchar>(j, i)=  img.at<uchar>(min_y + j - size_buffer/2, min_x + i - size_buffer/2);
		}
	}
	for(int i=0 ; i<contour.size() ; i++){
		contour[i].x -= min_x-size_buffer/2;
		contour[i].y -= min_y-size_buffer/2;
	}
	return object;
}

void GUI::copyLightField(Mat **imgs, Mat **&out_imgs, Mat **imgs_depth, Mat **&out_imgs_depth, int u, int v)
{
	//out_imgs = new Mat*[u];
	//out_imgs_depth = new Mat*[u];
	for(int i=0 ; i<u ; i++){
		//out_imgs[i] = new Mat[v];
		//out_imgs_depth[i] = new Mat[v];
		for(int j=0 ; j<v ; j++){
			out_imgs[i][j] = imgs[i][j].clone();
			out_imgs_depth[i][j] = imgs_depth[i][j].clone();
		}
	}
}

void GUI::randomLightField(Mat **&out_imgs, Mat **&out_imgs_depth, Mat **&out_imgs_mask, int u, int v, int row, int col)
{
	out_imgs = new Mat*[u];
	out_imgs_depth = new Mat*[u];
	out_imgs_mask = new Mat*[u];
	for(int i=0 ; i<u ; i++){
		out_imgs[i] = new Mat[v];
		out_imgs_depth[i] = new Mat[v];
		out_imgs_mask[i] = new Mat[v];
		for(int j=0 ; j<v ; j++){
			out_imgs[i][j] = Mat(cvSize(col, row), CV_8UC3);
			out_imgs_depth[i][j] = Mat(cvSize(col, row), CV_8UC1);
			out_imgs_mask[i][j] = Mat::zeros(cvSize(col, row), CV_8UC1);
			for(int y=0 ; y<row ; y++)
				for(int x=0 ; x<col ; x++){
					Scalar s(rand()%256, rand()%256, rand()%256);
					//std::cout << "yoyoyoy" <<std::endl;
					out_imgs[i][j].at<Vec3b>(y, x)[0] = s[0];
					out_imgs[i][j].at<Vec3b>(y, x)[1] = s[1];
					out_imgs[i][j].at<Vec3b>(y, x)[2] = s[2];
					//std::cout << "yoyoyoy" <<std::endl;
					out_imgs_depth[i][j].at<uchar>(y, x) = s[0];
					//std::cout << "yoyoyoy" <<std::endl;
				}
		}
	}
}

void GUI::dispalyImg(Mat &src, Mat &tar, Mat &main_img, ImgType type)
{
	int gap = 30;
	if(type==COLOR){
		main_img = Mat(cvSize(src.cols + tar.cols + gap, src.rows), CV_8UC3);
		for(int i=0 ; i<tar.rows ; i++)
			for(int j=0 ; j<tar.cols ; j++){
				main_img.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
				main_img.at<Vec3b>(i, src.cols + gap + j) = tar.at<Vec3b>(i, j);
			}
	}else{
		main_img = Mat(cvSize(src.cols + tar.cols + gap, src.rows), CV_8UC1);
		for(int i=0 ; i<tar.rows ; i++)
			for(int j=0 ; j<tar.cols ; j++){
				main_img.at<uchar>(i, j) = src.at<uchar>(i, j);
				main_img.at<uchar>(i, src.cols + gap + j) = tar.at<uchar>(i, j);
			}
	}
	//imshow("><", main_img);
	//waitKey(0);
	//main_img = pair(Rect());
}


void GUI::transferAndPaste(Mat **src, Mat **src_depth, Mat **tar, Mat **tar_depth, Mat **tar_mask, int u, int v, 
					  Mat &object, vector<Point2f> &object_contour, Point2f shift)
{
	Mat descriptors_object;
	std::vector<KeyPoint> keypoints_object;
	Mat object_mask;
	int img_col = src[0][0].cols, img_row = src[0][0].rows;
	mask::find_mask_by_contour(object_contour, object.cols, object.rows, object_mask);
	find_features(object, object_mask, keypoints_object, descriptors_object, F_SURF);
//#pragma omp parallel for
	for(int i=0 ; i<u ; i++){
		for(int j=0 ; j<v ; j++){
			Mat descriptors_scene;
			std::vector<KeyPoint> keypoints_scene;
			// find features and descriptors
			find_features(src[i][j], Mat(), keypoints_scene, descriptors_scene, F_SURF);
			// match all the feature pairs
			FeaturePair feature_pair;
			std::vector<DMatch> good_matches;
			match_feature(feature_pair, keypoints_object, keypoints_scene, descriptors_object, descriptors_scene, good_matches);
			// find homography and transform mask
			Mat H = findHomography( feature_pair.obj, feature_pair.scene, CV_RANSAC );
			vector<Point2f> scene_contour, scene_contour2;
			perspectiveTransform( object_contour, scene_contour, H);
			perspectiveTransform( object_contour, scene_contour2, H);
			// copy and paste
			Point2f obj_position;
			Mat *obj = object_mask_gen(src[i][j], scene_contour, obj_position, COLOR);
			Mat *obj_depth = object_mask_gen(src_depth[i][j], scene_contour2, obj_position, DEPTH);
			Mat obj_mask;
			mask::find_mask_by_contour(scene_contour, obj->cols, obj->rows, obj_mask);
			paste2Target(tar[i][j], tar_depth[i][j], tar_mask[i][j], *obj, *obj_depth, obj_mask, obj_position+shift);
			std::cout << "img_" << i << "_" << j << " done" << std::endl;
			// for debug
			//draw_matching(img_object, keypoints_object, imgs[i][j], keypoints_scene, 
			//				good_matches, H, object_mask, imgs_mask[i][j]);
			
		}
	}
}


void removeRegion(Mat &img, Mat &depth, Mat &mask)
{
	for(int i=0 ; i<img.rows ; i++)
		for(int j=0 ; j<img.cols ; j++)
			if(mask.at<uchar>(i, j)>100){
				img.at<Vec3b>(i, j)[0] = rand()%256;
				img.at<Vec3b>(i, j)[1] = rand()%256;
				img.at<Vec3b>(i, j)[2] = rand()%256;
				depth.at<uchar>(i, j) = rand()%256;
			}
}

void GUI::deleteRegion(Mat **src, Mat **src_depth, Mat **tar, Mat **tar_depth, int u, int v, 
					  Mat &object, vector<Point2f> &object_contour)
{
	Mat descriptors_object;
	std::vector<KeyPoint> keypoints_object;
	Mat object_mask;
	int img_col = src[0][0].cols, img_row = src[0][0].rows;
	mask::find_mask_by_contour(object_contour, object.cols, object.rows, object_mask);
	find_features(object, object_mask, keypoints_object, descriptors_object, F_SURF);
//#pragma omp parallel for
	for(int i=0 ; i<u ; i++){
		for(int j=0 ; j<v ; j++){
			Mat descriptors_scene;
			std::vector<KeyPoint> keypoints_scene;
			// find features and descriptors
			find_features(src[i][j], Mat(), keypoints_scene, descriptors_scene, F_SURF);
			// match all the feature pairs
			FeaturePair feature_pair;
			std::vector<DMatch> good_matches;
			match_feature(feature_pair, keypoints_object, keypoints_scene, descriptors_object, descriptors_scene, good_matches);
			// find homography and transform mask
			Mat H = findHomography( feature_pair.obj, feature_pair.scene, CV_RANSAC );
			vector<Point2f> scene_contour;
			perspectiveTransform( object_contour, scene_contour, H);
			Mat img_mask;
			mask::find_mask_by_contour(scene_contour, img_col, img_row, img_mask);
			removeRegion(tar[i][j], tar_depth[i][j], img_mask);
			std::cout << "img_" << i << "_" << j << " done" << std::endl;
		}
	}
}
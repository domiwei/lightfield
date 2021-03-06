
#include "opencv2/features2d/features2d.hpp"

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "mask.hpp"
#include <iostream>
using namespace cv;

#ifndef __MASK_H__
#define __MASK_H__
/*feature type*/
typedef enum{
	F_SURF, F_SIFT
}Feature_Type;

/*matching pair*/
struct FeaturePair{
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
};

static void find_features(Mat &img, Mat &mask, std::vector<KeyPoint> &keypoints, Mat &descriptors, Feature_Type type)
{
	if(type==F_SURF){
		//-- Step 1: Detect the keypoints using SURF Detector
		int minHessian = 400;
		SurfFeatureDetector detector( minHessian );
		detector.detect( img, keypoints, mask );
		//-- Step 2: Calculate descriptors (feature vectors)
		SurfDescriptorExtractor extractor;
		extractor.compute( img, keypoints, descriptors );

	}
}

static void match_feature(FeaturePair &feature_pair, std::vector<KeyPoint> keypoints_object,
					std::vector<KeyPoint> keypoints_scene, Mat &descriptors_object, 
					Mat &descriptors_scene, std::vector<DMatch> &good_matches)
{
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_object, descriptors_scene, matches );
	double max_dist = 0; double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_object.rows; i++ ){ 
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	//printf("-- Max dist : %f \n", max_dist );
	//printf("-- Min dist : %f \n", min_dist );
	 //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	//std::vector< DMatch > good_matches;
	for( int i = 0; i < descriptors_object.rows; i++ )
		if( matches[i].distance < (0.5*min_dist + 0.5*max_dist) )
			good_matches.push_back( matches[i]);
	//-- Localize the object
	for( int i = 0; i < good_matches.size(); i++ ){
		//-- Get the keypoints from the good matches
		feature_pair.obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
		feature_pair.scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
	}
	return ;
}



static void draw_matching(Mat &img_object, std::vector<KeyPoint> &keypoints_object, 
					Mat &img_scene, std::vector<KeyPoint> &keypoints_scene,
					std::vector<DMatch> good_matches, Mat &H, Mat &object_mask, Mat &img_mask)
{
	Mat img_matches;
	std::vector<KeyPoint> non1, non2;
	std::vector<DMatch> match;
	drawMatches( img_object, non1, img_scene, non2,
               match, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);
  perspectiveTransform( obj_corners, scene_corners, H);
  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
  line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  for(int i=0 ; i<object_mask.cols ; i++)
	  for(int j=0 ; j<object_mask.rows ; j++)
		  if(object_mask.at<uchar>(j, i)==255)
			  img_matches.at<cv::Vec3b>(j, i)[0] = 255;
  for(int i=0 ; i<img_mask.cols ; i++)
	  for(int j=0 ; j<img_mask.rows ; j++)
		  if(img_mask.at<uchar>(j, i)==255)
			  img_matches.at<cv::Vec3b>(j , i + img_object.cols)[0] = 255;	  
  //-- Show detected matches
  imshow( "Good Matches & Object detection", img_matches );
  waitKey(0);
  return ;
}

/* main function for generating masks */
static void generate_lightfield_mask(Mat **imgs, Mat **imgs_mask, int row, int column, 
								Mat &img_object, vector<Point2f> object_point)
{
	Mat descriptors_object;
	std::vector<KeyPoint> keypoints_object;
	Mat object_mask;
	int img_col = imgs[0][0].cols, img_row = imgs[0][0].rows;
	mask::find_mask_by_contour(object_point, img_object.cols, img_object.rows, object_mask);
	find_features(img_object, object_mask, keypoints_object, descriptors_object, F_SURF);
#pragma omp parallel for
	for(int i=0 ; i<row ; i++){
		for(int j=0 ; j<column ; j++){
			Mat descriptors_scene;
			std::vector<KeyPoint> keypoints_scene;
			// find features and descriptors
			find_features(imgs[i][j], Mat(), keypoints_scene, descriptors_scene, F_SURF);
			// match all the feature pairs
			FeaturePair feature_pair;
			std::vector<DMatch> good_matches;
			match_feature(feature_pair, keypoints_object, keypoints_scene, descriptors_object, descriptors_scene, good_matches);
			// find homography and transform mask
			Mat H = findHomography( feature_pair.obj, feature_pair.scene, CV_RANSAC );
			vector<Point2f> scene_contour;
			perspectiveTransform( object_point, scene_contour, H);
			// draw the final mask
			mask::find_mask_by_contour(scene_contour, img_col, img_row, imgs_mask[i][j]);
			std::cout << "img_" << i << "_" << j << " done" << std::endl;
			// for debug
			//draw_matching(img_object, keypoints_object, imgs[i][j], keypoints_scene, 
			//				good_matches, H, object_mask, imgs_mask[i][j]);
			
		}
	}
  //Mat img_object = imread( object, CV_LOAD_IMAGE_COLOR );
  //Mat img_scene = imread( "test/8_8.jpg", CV_LOAD_IMAGE_COLOR );
  //-- Step 3: Matching descriptor vectors using FLANN matcher
}


#endif
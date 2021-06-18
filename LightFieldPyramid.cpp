#include "LightFieldPyramid.h"

void imgDownSample(Mat &src, Mat &tar, Mat &mask_src, Mat &mask_tar, Mat &depth_src, Mat &depth_tar)  //down sample 50%
{
	int row = src.rows;
	int col = src.cols;
	for(int y=0 ; y<row-2 ; y+=2){
		for(int x=0 ; x<col-2 ; x+=2){
			//std::cout << y <<", " << x << std::endl;
			tar.at<Vec3b>(y/2, x/2) = src.at<Vec3b>(y, x)/4+
									  src.at<Vec3b>(y+1, x)/4+
									  src.at<Vec3b>(y, x+1)/4+
									  src.at<Vec3b>(y+1, x+1)/4;
			if(&mask_src && &mask_tar)
				mask_tar.at<uchar>(y/2, x/2) = mask_src.at<uchar>(y, x);
			depth_tar.at<uchar>(y/2, x/2) = depth_src.at<uchar>(y, x);
		}
	}
}

void imgDownSample(Mat &src, Mat &tar, Mat &depth_src, Mat &depth_tar)  //down sample 50%
{
	int row = src.rows;
	int col = src.cols;
	for(int y=0 ; y<row-2 ; y+=2){
		for(int x=0 ; x<col-2 ; x+=2){
			//std::cout << y <<", " << x << std::endl;
			tar.at<Vec3b>(y/2, x/2) = src.at<Vec3b>(y, x)/4+
									  src.at<Vec3b>(y+1, x)/4+
									  src.at<Vec3b>(y, x+1)/4+
									  src.at<Vec3b>(y+1, x+1)/4;
			depth_tar.at<uchar>(y/2, x/2) = depth_src.at<uchar>(y, x);
		}
	}
}

LightField *lightfieldDownSample(const LightField lf)
{
	Mat **imgs = new Mat*[lf.u];
	Mat **imgs_mask = new Mat*[lf.u];
	Mat **imgs_depth = new Mat*[lf.u];
	for(int i=0 ; i<lf.u ; i++){
		imgs[i] = new Mat[lf.v];
		imgs_mask[i] = new Mat[lf.v];
		imgs_depth[i] = new Mat[lf.v];
		for(int j=0 ; j<lf.v ; j++){
			/*imgs[i][j] = Mat(lf.row/2, lf.col/2, CV_8UC3);
			imgs_mask[i][j] = Mat(lf.row/2, lf.col/2, CV_8UC1);
			imgs_depth[i][j] = Mat(lf.row/2, lf.col/2, CV_8UC1);*/
			resize(lf.cvimg[i][j], imgs[i][j], Size(lf.cvimg[i][j].cols/2.0, lf.cvimg[i][j].rows/2.0)); 
			//imgs_mask[i][j] = Mat(lf.row/2, lf.col/2, CV_8UC1);
			resize(lf.cvdepth[i][j], imgs_depth[i][j], Size(lf.cvdepth[i][j].cols/2.0, lf.cvdepth[i][j].rows/2.0));
			//if(lf.cvmask)
			resize(lf.cvmask[i][j], imgs_mask[i][j], Size(lf.cvmask[i][j].cols/2.0, lf.cvmask[i][j].rows/2.0));
			//std::cout << "lalala" << (lf.cvmask) <<std::endl;
		//	if(lf.cvmask)
		//		imgDownSample(lf.cvimg[i][j], imgs[i][j], lf.cvmask[i][j], imgs_mask[i][j], lf.cvdepth[i][j], imgs_depth[i][j]);
		//	else
		//		imgDownSample(lf.cvimg[i][j], imgs[i][j], lf.cvdepth[i][j], imgs_depth[i][j]);
			//std::cout << "ya" <<std::endl;
		}
	}
	LightField *new_lf = new LightField(imgs, imgs_mask, lf.u, lf.v, imgs_depth);
	return new_lf;
}

void LightFieldPyramid::pyramidInitial(LightField lf)
{
	pyramid.push_back(&lf);
	for(int i=1 ; i<layer ; i++){
		LightField *down_sample_lf = lightfieldDownSample(*pyramid[i-1]);
		//imshow("tmp", down_sample_lf->cvmask);
		pyramid.push_back(down_sample_lf);
		std::cout << "row, col = "<< down_sample_lf->row << ", "<< down_sample_lf->col << std::endl;
	}
}

void setLayer(LightField &lf, int layer)
{

}

LightFieldPyramid::~LightFieldPyramid()
{
	
}




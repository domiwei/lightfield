#include "4DPatchMatch.h"
#include "opencv2/imgproc/imgproc.hpp"
typedef enum{
	TRUST_X, TRUST_Y
}Trust;
#define MAX_DEPTH 2 // 6 for beans, 2 for butterfly
#define DEPTH_SAMPLE 64
class DepthEstimator{
public:
	DepthEstimator(Point4D size){
		depth_map = new Mat*[size.u];
		for(int i=0 ; i<size.u ; i++){
			depth_map[i] = new Mat[size.v];
			for(int j=0 ; j<size.v ; j++){
				depth_map[i][j] = Mat::zeros(size.y, size.x, CV_32FC1);
			}
		}
	};
	void estimate(LightFieldData<uchar, 3> &lf)
	{ 
		Mat img = Mat(lf.size.y, lf.size.x, CV_8UC3);
		for(int u=0 ; u<lf.size.u ; u++)
			for(int v=0 ; v<lf.size.v ; v++){
				for(int y=0 ; y<lf.size.y ; y++){
					std::cout << "(u, v, y, x) = (" << u << ", " << v << ", " << y << ")" << std::endl;
					for(int x=0 ; x<lf.size.x ; x++){
						//float conx, cony;
						if(depth_map[u][v].at<float>(y, x)!=0.0)
							continue;
						float conx, cony;
						Point4D p(u,v,y,x);
						confidenceEstimator(lf, p, conx, cony);
						/*if(conx<0.02 && cony<0.02){
							depth_map[u][v].at<float>(y, x) = -MAX_DEPTH;
							continue;
						}*/
						Trust trust = conx>cony?TRUST_X:TRUST_Y;
						float depth = depthEstimator(lf, p, trust);
					//	depth_map[u][v].at<float>(y, x) = depth;
						depthPropagate(p, depth, lf);
					}
				}
			}
		//imshow("depth", img);
		waitKey(0);
	}
	Mat showDepthMap(int u, int v)
	{
		double min;
		double max;
		cv::minMaxIdx(depth_map[u][v], &min, &max);
		Mat *img = new Mat(depth_map[u][v].rows, depth_map[u][v].cols, CV_8UC1);
		for(int y=0 ; y<depth_map[u][v].rows ; y++){
			for(int x=0 ; x<depth_map[u][v].cols ; x++){
				float depth = (depth_map[u][v].at<float>(y, x)+MAX_DEPTH)/(2*MAX_DEPTH)*255;
				img->at<uchar>(y, x) = depth;
		//		std::cout << depth << std::endl;
			}
		}
		//imshow("depth", img);
				//waitKey(0);
		medianBlur(*img, *img, 5);
		char filename[20];
		sprintf(filename, "depth_%d_%d.bmp", u, v);
		imwrite(filename, *img);
		return *img;
		//imwrite("depth_median.bmp", img);
		//imshow("depth", img);
		//waitKey(0);
	}
private:
	Mat **depth_map;
	void confidenceEstimator(LightFieldData<uchar, 3> &lf, Point4D p, float &conx, float &cony);
	float depthEstimator(LightFieldData<uchar, 3> &lf, Point4D p, Trust trust);
	void depthPropagate(Point4D p, float depth, LightFieldData<uchar, 3> &lf);
};
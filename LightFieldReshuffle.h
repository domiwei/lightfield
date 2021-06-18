#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "LightFieldPyramid.h"
#include <iostream>     /* srand, rand */
#include <time.h>       /* time */
//#include <cmath>

LightField *lightfieldResize(LightField *&now, int col, int row);
static LightField *initialTarget(LightField *&now, LightField *next);

class LightFieldReshuffle{
public:
	LightFieldReshuffle(){}
	LightField &reshuffling(LightField &src_lf, LightField &tar_lf,
							Point4D patch_size, int layer = 1)
	{
		y_scale = 1;
		x_scale = 1;
		this->patch_size = patch_size;
		LightFieldPyramid lfp(layer);  
		lfp.pyramidInitial(src_lf); // set pyramid
		std::cout << "generating pyramid done!" <<std::endl;
		//LightField *scale_lf = new LightField(lf.cvimg, lf.cvmask, lf.u, lf.v, lf.cvdepth);
		//for(int i=0 ; i<resize_times ; i++){
			//scale_lf = directScaleLF(scale_lf, y_scale, x_scale);
			std::cout << "begin to reshuffle " <<std::endl;
			//imshow("tmp", scale_lf->cvimg[0][0]);
			//waitKey(0);
			LightFieldPyramid scale_lfp(layer);  
			scale_lfp.pyramidInitial(tar_lf); // set pyramid
			src_map = NULL;
			tar_map = NULL;
			LightField *now = scale_lfp.getLayer(layer-1);
			for(int l=layer-1 ; l>=0 ; l--){
				std::cout << "--------------layer " << l << "--------------" << std::endl;
				//char file[20];
				//sprintf(file, "resize_fly/w_%d_%d_direc_scale.jpg", i, layer-l-1);
				//scale_lfp.getLayer(l)->writeImage(file, 0, 0);
				for(int r=0 ; r<tar_lf.u ; r++)
					for(int c=0 ; c<tar_lf.v ; c++){
						char out_name[30];
						sprintf(out_name, "reshuffle5/before_layer%d_%d_%d.jpg", layer-l-1, r, c);
						now->writeImage(&out_name[0], r, c);
				}
				
				Map **ref_src_map = src_map;
				Map **ref_tar_map = tar_map;
				PatchMatchRetarget(*lfp.getLayer(l), *now, ref_src_map, ref_tar_map, 4, l);
				
				//PatchMatchRetarget(lf, *scale_lf, ref_src_map, ref_tar_map);
				//char file2[20];
				//sprintf(file2, "resize_fly/w_%d_%d.jpg", i, layer-l-1);
				//now->writeImage(file2, 0, 0);
				/*
				if(l>0){
					//char file[20];
					//sprintf(file, "wahaha.jpg");
					//now->writeImage(file, 0, 0);
					now = lightfieldResize(now, scale_lfp.getLayer(l-1)->col, scale_lfp.getLayer(l-1)->row);
					//sprintf(file, "wahaha2.jpg");
					//now->writeImage(file, 0, 0);
					//system("pause");
				}else{
					scale_lf = lightfieldResize(now, scale_lfp.getLayer(0)->col, scale_lfp.getLayer(0)->row);
				}
				*/
				//std::cout << now->u << ", " << now->v << std::endl;
				//////////////////
				for(int r=0 ; r<tar_lf.u ; r++)
					for(int c=0 ; c<tar_lf.v ; c++){
						char out_name[30];
						sprintf(out_name, "reshuffle5/layer%d_%d_%d.jpg", layer-l-1, r, c);
						now->writeImage(&out_name[0], r, c);
				}
				if(l>0){
					//now = initialTarget(now, scale_lfp.getLayer(l-1));
					now = scale_lfp.getLayer(l-1);
				}
				//////////////////
			}
			
		//}
		return tar_lf;
	
	}

private:
	float y_scale;
	float x_scale;
	Map **src_map;
	Map **tar_map;
	Point4D patch_size;
	LightField *directScaleLF(LightField *lf, float y_scale, float x_scale);
	void PatchMatchRetarget(LightField &lf, LightField &scale_lf, Map **ref_src_map, Map **ref_tar_map, int itration, int layer);
	void initial(LightField &lf, LightField &scale_lf, LightFieldData<float, 1> *min_table, Map **ref_map, Map **self_map);
	void propagate_and_randomSearch(LightField &lf, LightField &scale_lf, LightFieldData<float, 1> *&min_table, Map **self_map, int phase,
									LightFieldData<float, 4> &weight_table, LightFieldData<float, 1> &depth_table);
	void refine(LightField &scale_lf, LightFieldData<float, 4> &weight_table, LightFieldData<float, 1> &depth_table,
				LightFieldData<float, 4> &weight_table_com, LightFieldData<float, 1> &depth_table_com, float scaling_rate);
	//
	Point4D get_src_shift(Point4D p){ return src_map[p.u][p.v].get_value(p.y, p.x); }
	void set_src_shift(Point4D &p, Point4D &shift){ src_map[p.u][p.v].set_value(p.y, p.x, shift); }
	Point4D get_tar_shift(Point4D p){ return tar_map[p.u][p.v].get_value(p.y, p.x); }
	void set_tar_shift(Point4D &p, Point4D &shift){ tar_map[p.u][p.v].set_value(p.y, p.x, shift); }
	Point4D get_shift(Point4D p, Map **map){ return map[p.u][p.v].get_value(p.y, p.x); }
	void set_shift(Point4D &p, Point4D &shift, Map **map){ map[p.u][p.v].set_value(p.y, p.x, shift); }
};


static LightField *initialTarget(LightField *&now, LightField *next)
{
	//assert(now->u < 10 && now->v <10);
	LightField *tmplf = lightfieldResize(now, next->col, next->row);
	//return tmplf;
	Mat **imgs = new Mat*[next->u];
	Mat **imgs_depth = new Mat*[next->u];
	for(int i=0 ; i<next->u ; i++){
		imgs[i] = new Mat[next->v];
		imgs_depth[i] = new Mat[next->v];
		for(int j=0 ; j<next->v ; j++){
			imgs[i][j] = Mat(next->row, next->col, CV_8UC3);
			imgs_depth[i][j] = Mat(next->row, next->col, CV_8UC1);
			for(int y=0 ; y<next->row ; y++){
				for(int x=0 ; x<next->col ; x++){
					//Scalar color = tmplf->img->get_value(Point4D(i, j, y, x));
					if(next->in_mask(Point4D(i, j, y, x))){
						imgs[i][j].at<Vec3b>(y, x) = next->cvimg[i][j].at<Vec3b>(y, x);
						imgs_depth[i][j].at<uchar>(y, x) = next->cvdepth[i][j].at<uchar>(y, x);
					}else{
						imgs[i][j].at<Vec3b>(y, x) = tmplf->cvimg[i][j].at<Vec3b>(y, x);
						imgs_depth[i][j].at<uchar>(y, x) = tmplf->cvdepth[i][j].at<uchar>(y, x);
					}
					//imgs[i][j].at<Vec3b>(y, x)[0] = (color[0] + color2[0])/2;
					//imgs[i][j].at<Vec3b>(y, x)[1] = (color[1] + color2[1])/2;
					//imgs[i][j].at<Vec3b>(y, x)[2] = (color[2] + color2[2])/2;
					//imgs_depth[i][j].at<uchar>(y, x) = (tmplf->depth->get_value(Point4D(i, j, y, x))[0]+
						//								next->depth->get_value(Point4D(i, j, y, x))[0])/2;
				}
			}		
		}
	}
	LightField *ret = new LightField(imgs, NULL, now->u, now->v, imgs_depth); 
	return ret;
}

static LightField *lightfieldResize(LightField *&now, int col, int row)
{
	//std::cout << now->u << ", " << now->v <<std::endl;
	assert(now->u < 10 && now->v <10);
	Mat **imgs = new Mat*[now->u];
	Mat **imgs_depth = new Mat*[now->u];
	for(int i=0 ; i<now->u ; i++){
		imgs[i] = new Mat[now->v];
		imgs_depth[i] = new Mat[now->v];
		for(int j=0 ; j<now->v ; j++){
			imgs[i][j] = Mat(now->row, now->col, CV_8UC3);
			imgs_depth[i][j] = Mat(now->row, now->col, CV_8UC1);
			for(int y=0 ; y<now->row ; y++){
				for(int x=0 ; x<now->col ; x++){
					Scalar color = now->img->get_value(Point4D(i, j, y, x));
					imgs[i][j].at<Vec3b>(y, x)[0] = color[0];
					imgs[i][j].at<Vec3b>(y, x)[1] = color[1];
					imgs[i][j].at<Vec3b>(y, x)[2] = color[2];
					imgs_depth[i][j].at<uchar>(y, x) = now->depth->get_value(Point4D(i, j, y, x))[0];
				}
			}
			resize(imgs[i][j], imgs[i][j], Size(col, row));
			resize(imgs_depth[i][j], imgs_depth[i][j], Size(col, row));
		}
	}
	LightField *ret = new LightField(imgs, NULL, now->u, now->v, imgs_depth); 
	return ret;
}
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "LightFieldPyramid.h"
#include <iostream>     /* srand, rand */
#include <time.h>       /* time */
//#include <cmath>

LightField *lightfieldResize(LightField *&now, int col, int row);


class LightFieldRetarget{
public:
	LightFieldRetarget(){}
	LightField *retargeting(LightField &lf, float vertical_scale, float horizontal_scale,
					 Point4D patch_size, int layer = 1)
	{
		int resize_times = 5;
		y_scale = pow((double)vertical_scale, (double)1.0/resize_times);
		x_scale = pow((double)horizontal_scale, (double)1.0/resize_times);
		this->patch_size = patch_size;
		LightFieldPyramid lfp(layer);  
		lfp.pyramidInitial(lf); // set pyramid
		std::cout << "generating pyramid done!" <<std::endl;
		LightField *scale_lf = new LightField(lf.cvimg, lf.cvmask, lf.u, lf.v, lf.cvdepth);
		for(int i=0 ; i<resize_times ; i++){
			scale_lf = directScaleLF(scale_lf, y_scale, x_scale);
			std::cout << "begin to retarget " <<std::endl;
			//imshow("tmp", scale_lf->cvimg[0][0]);
			//waitKey(0);
			LightFieldPyramid scale_lfp(layer);  
			scale_lfp.pyramidInitial(*scale_lf); // set pyramid
			src_map = NULL;
			tar_map = NULL;
			LightField *now = scale_lfp.getLayer(layer-1);
			for(int l=layer-1 ; l>=0 ; l--){
				char file[20];
				sprintf(file, "resize_fly_home/%d_%d_direc_scale.jpg", i, layer-l-1);
				scale_lfp.getLayer(l)->writeImage(file, 0, 0);
				Map **ref_src_map = src_map;
				Map **ref_tar_map = tar_map;
				PatchMatchRetarget(*lfp.getLayer(l), *now, ref_src_map, ref_tar_map, 2+l*2, l);
				//PatchMatchRetarget(lf, *scale_lf, ref_src_map, ref_tar_map);
				char file2[20];
				sprintf(file2, "resize_fly_home/%d_%d.jpg", i, layer-l-1);
				//////////////////
				for(int r=0 ; r<scale_lf->u ; r++)
					for(int c=0 ; c<scale_lf->v ; c++){
						char out_name[30];
						sprintf(out_name, "resize_fly_home/times%d_layer%d_%d_%d.jpg", i, layer-l-1, r, c);
						if(l>0)
							now->writeImage(&out_name[0], r, c);
						else
							scale_lf->writeImage(&out_name[0], r, c);
				}
				//////////////////
				now->writeImage(file2, 0, 0);
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
				//std::cout << now->u << ", " << now->v << std::endl;
				
			}
			char file[20];
			sprintf(file, "resize_fly_home/%d_%d_origin.jpg", i, layer-1);
			lfp.getLayer(0)->writeImage(file, 0, 0);
			
		}
		return scale_lf;
		/*
		for(int i=0 ; i<lf.u ; i++)
			for(int j=0 ; j<lf.v ; j++){
				char out_name[30];
				sprintf(out_name, "out3/%d_%d.jpg", i, j);
			//std::cout << out_name << std::endl;
				scale_lf->writeImage(&out_name[0], i, j);
			//imwrite(filename, lf.img[i][j]);
			
			}
		return new LightField(scale_lf->cvimg, scale_lf->cvmask, lf.u, lf.v, scale_lf->cvmask);
		*/
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



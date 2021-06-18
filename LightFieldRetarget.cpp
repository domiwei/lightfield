#include "LightFieldRetarget.h"
#include "opencv2/imgproc/imgproc.hpp"
extern void getNewMapAndMinTable(Map **&map, LightFieldData<float, 1> *&min_table, Point4D size);
extern Point4D getRandomPos(LightField &lf, Point4D base, Point4D min, Point4D max, MASK_FLAG mask);
//extern void setWeightTable(LightField &lf, Point4D &src, Point4D &tar, 
//					const double weight, const Point4D &patch_size, 
//					LightFieldData<float, 4> &table, LightFieldData<float, 1> &depth_table);
extern std::ostream &operator<<(std::ostream &out, const Point4D &p);
#define MAX_DEPTH 2 //2
int max_depth;

LightField *lightfieldResize(LightField *&now, int col, int row)
{
	std::cout << now->u << ", " << now->v <<std::endl;
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


/* used for computing patch difference */
inline float length(Scalar s)
{
	float dis = 0.0;
	for(int i=0 ; i<3 ; i++)
		dis += s[i]*s[i];
	return (dis);
}

void setWeightTable(LightField &scale_lf, LightField &lf, Point4D &src, Point4D &tar, 
					const double weight, const Point4D &patch_size, LightFieldData<float, 4> &table, LightFieldData<float, 1> &depth_table)
{
	for(int u=-patch_size.u ; u<=patch_size.u ; u++)
		for(int v=-patch_size.v ; v<=patch_size.v ; v++)
			for(int y=-patch_size.y ; y<=patch_size.y ; y++)
				for(int x=-patch_size.x ; x<=patch_size.x ; x++){
					Point4D shift = Point4D(u,v,y,x);
					Point4D s = src+shift;
					Point4D t = tar+shift;
					if(!(scale_lf.check_4D_bound(s) && lf.check_4D_bound(t)))
						continue;
					Point4D tsize = table.size;
					//if(s.y >= tsize.y || s.x >= tsize.x || s.y<0 || s.x<0)
					//	std::cout << tsize <<std::endl;
					Scalar data = table.get_value(s);
					data[3] += weight;
					Scalar color = lf.get_value(t, NON_MASK);
					for(int ch=0 ; ch<3 ; ch++)
						data[ch] += (color[ch]*weight);
					table.set_value(s, data);

					double d = depth_table.get_value(s)[0] + lf.getDepth(t, NON_MASK)*weight;
					depth_table.set_value(s, Scalar(d));
				}
}




double patch_diff(LightField &lf1, LightField &lf2, Point4D p1, 
					Point4D p2, Point4D patch_size)
{
#define LAMBDA 0.2  // depth weight
#define BETA 0  // consistency weight
//#define NOT_IN_MASK_WEIGHT 16
//#define IN_MASK_WEIGHT 1

	int count = 0;
	double diff = 0;
	double depth_diff = 0;  // for depth
	for(int u=-patch_size.u ; u<=patch_size.u ; u++)
		for(int v=-patch_size.v ; v<=patch_size.v ; v++)
			for(int y=-patch_size.y ; y<=patch_size.y ; y++)
				for(int x=-patch_size.x ; x<=patch_size.x ; x++){
					if(!lf1.check_4D_bound(p1 + Point4D(u,v,y,x)) || !lf2.check_4D_bound(p2 + Point4D(u,v,y,x)))
						continue;
					Scalar color1 = lf1.get_value(p1 + Point4D(u,v,y,x), NON_MASK);
					Scalar color2 = lf2.get_value(p2 + Point4D(u,v,y,x), NON_MASK);
					uchar depth1 = lf1.getDepth(p1 + Point4D(u,v,y,x), NON_MASK); // for depth
					uchar depth2 = lf2.getDepth(p2 + Point4D(u,v,y,x), NON_MASK); // for depth
				//	if(color1[0]<0 || color2[0]<0 || depth1<0 ||depth2<0)
				//		std::cout << depth1 << ", " << depth2 <<std::endl;
					int w1=1, w2 = 1;
					//int w1 = lf1.in_mask(p1 + Point4D(u,v,y,x))? IN_MASK_WEIGHT : NOT_IN_MASK_WEIGHT;
					//int w2 = lf2.in_mask(p2 + Point4D(u,v,y,x))? IN_MASK_WEIGHT : NOT_IN_MASK_WEIGHT;
					double color_diff = 0;
					for(int ch=0 ; ch<3 ; ch++)
						color_diff += (color1[ch]-color2[ch])*(color1[ch]-color2[ch]);
					diff += (color_diff/255/255/3*w1*w2);
					depth_diff += (((double)depth1-(double)depth2)*((double)depth1-(double)depth2)/255/255*w1*w2);
					count += (w1*w2);
				}
	double intensity =  ((1-LAMBDA)*(diff/count) + LAMBDA*depth_diff/count);
	return intensity;
	/////////////////////////////////////////////////////////////////////////////
	
	Scalar color = lf2.get_value(p2, NON_MASK);
	double p2_depth = lf2.getDepth(p2, NON_MASK);
	double depth = lf1.getDepth(p1, NON_MASK); // for depth
	double disparity = depth/255*2*max_depth - max_depth;
	double error = 0;
	double depth_error = 0;
	for(int v=0 ; v<lf1.v ; v++){ //h
		float position = p1.x + (p1.v - v)*disparity;
		int int_position = floor(position);
		Scalar color1 = lf1.get_value(Point4D(p1.u, v, p1.y, int_position), NON_MASK);
		Scalar color2 = lf1.get_value(Point4D(p1.u, v, p1.y, int_position+1), NON_MASK);
		if(color1[0]<0 || color2[0]<0)
			continue;
		Scalar interpolated_color = (color2*(position-int_position) + color1*(int_position+1-position));
		error += (length(color - interpolated_color)/255/255/3);
		// depth consistency
		double depth1 = lf1.getDepth(Point4D(p1.u, v, p1.y, int_position), NON_MASK);
		double depth2 = lf1.getDepth(Point4D(p1.u, v, p1.y, int_position+1), NON_MASK);
		double interpolated_depth = (depth2*(position-int_position) + depth1*(int_position+1-position));
		depth_error += ((interpolated_depth - p2_depth)*(interpolated_depth - p2_depth)/255/255);
	}
	for(int u=0 ; u<lf1.u ; u++){ //v
		float position = p1.y + (p1.u - u)*(-disparity);
		int int_position = floor(position);
		Scalar color1 = lf1.get_value(Point4D(u, p1.v, int_position, p1.x), NON_MASK);
		Scalar color2 = lf1.get_value(Point4D(u, p1.v, int_position+1, p1.x), NON_MASK);
		if(color1[0]<0 || color2[0]<0)
			continue;
		Scalar interpolated_color = (color2*(position-int_position) + color1*(int_position+1-position));
		error += (length(color - interpolated_color)/255/255/3);
		// depth consistency
		double depth1 = lf1.getDepth(Point4D(u, p1.v, int_position, p1.x), NON_MASK);
		double depth2 = lf1.getDepth(Point4D(u, p1.v, int_position+1, p1.x), NON_MASK);
		double interpolated_depth = (depth2*(position-int_position) + depth1*(int_position+1-position));
		depth_error += ((interpolated_depth - p2_depth)*(interpolated_depth - p2_depth)/255/255);
	}
	//double consistency = (1-LAMBDA)*error/(lf.u+lf.v-2) + LAMBDA*depth_error/(lf.u+lf.v-2);
	double consistency = 0.5*error/(lf1.u+lf1.v-2) + 0.5*depth_error/(lf1.u+lf1.v-2);
	consistency = (lf1.u+lf1.v-2==0 ? 0 : consistency);
	//return consistency;
	/////////////////////////////////////////////////////////////////////////////
	return (1-BETA)*intensity + BETA*consistency;
	
	
}



LightField *LightFieldRetarget::directScaleLF(LightField *lf, float y_scale, float x_scale)
{

	std::cout << lf->u  <<std::endl;
	assert(lf->u <10 && lf->v <10);
	//imshow("tmp", lf->cvimg[0][0]);
	//waitKey(0);
	Mat **imgs = new Mat*[lf->u];
	Mat **imgs_depth = new Mat*[lf->u];
	for(int i=0 ; i<lf->u ; i++){
		imgs[i] = new Mat[lf->v];
		imgs_depth[i] = new Mat[lf->v];
		for(int j=0 ; j<lf->v ; j++){
			imgs[i][j] = Mat();
			imgs_depth[i][j] = Mat();
			resize(lf->cvimg[i][j], imgs[i][j], Size(lf->col*x_scale, lf->row*y_scale));
			resize(lf->cvdepth[i][j], imgs_depth[i][j], Size(lf->col*x_scale, lf->row*y_scale));
		}
	}
	LightField *ret = new LightField(imgs, NULL, lf->u, lf->v, imgs_depth);
	return ret;
}

typedef enum {
	COMPLETE, COHERENCE 
}Direction;
Direction dir;
void LightFieldRetarget::PatchMatchRetarget(LightField &lf, LightField &scale_lf, Map **ref_src_map, Map **ref_tar_map, int itration, int layer)
{
	float scaling_rate = ((float)scale_lf.row/ lf.row)*((float)scale_lf.col/ lf.col);
	std::cout << "scaling_rate = " << scaling_rate << std::endl;
	LightFieldData<float, 1> *src_min_table;
	LightFieldData<float, 1> *tar_min_table;
	getNewMapAndMinTable(src_map, src_min_table, Point4D(lf.u, lf.v, lf.row, lf.col));
	getNewMapAndMinTable(tar_map, tar_min_table, Point4D(scale_lf.u, scale_lf.v, scale_lf.row, scale_lf.col));
	dir = COHERENCE;
	max_depth = MAX_DEPTH / pow(2.0, layer);  // Recapitalize the depth map
	initial(lf, scale_lf, tar_min_table, ref_tar_map, tar_map);  //initialize target from src
	std::cout << "initial done" <<std::endl;
	//scale_lf.writeImage("wahaha.jpg", 0, 0);
	//system("pause");
	dir = COMPLETE;
	initial(scale_lf, lf, src_min_table, ref_src_map, src_map);  //initialize src from target
	//lf.writeImage("wahaha.jpg", 0, 0);
	//system("pause");
	std::cout << "initial done" <<std::endl;
	for(int i=0 ; i<itration ; i++){
		LightFieldData<float, 4> weight_table_coh(Point4D(scale_lf.u, scale_lf.v, scale_lf.row, scale_lf.col));
		LightFieldData<float, 1> depth_table_coh(Point4D(scale_lf.u, scale_lf.v, scale_lf.row, scale_lf.col));
		dir = COHERENCE;
		propagate_and_randomSearch(lf, scale_lf, tar_min_table, tar_map, i%2, weight_table_coh, depth_table_coh);
		LightFieldData<float, 4> weight_table_com(Point4D(scale_lf.u, scale_lf.v, scale_lf.row, scale_lf.col));
		LightFieldData<float, 1> depth_table_com(Point4D(scale_lf.u, scale_lf.v, scale_lf.row, scale_lf.col));
		dir = COMPLETE;
		propagate_and_randomSearch(scale_lf, lf, src_min_table, src_map, i%2, weight_table_com, depth_table_com);
		refine(scale_lf, weight_table_coh, depth_table_coh, weight_table_com, depth_table_com, scaling_rate);
		//imshow("tmp", scale_lf.cvimg[0][0]);
		//waitKey(0);
	}
	//system("pause");
//	}
}


void LightFieldRetarget::initial(LightField &lf, LightField &scale_lf, LightFieldData<float, 1> *min_table, Map **ref_map, Map **self_map)
{
	std::cout << "begin to initial " <<std::endl;
	Point4D sample_range(2, 2, lf.row*0.05, lf.col*0.05);
	for(int u=0 ; u<scale_lf.u ; u++)
		for(int v=0 ; v<scale_lf.v ; v++){
			std::cout << u << ", " << v <<std::endl;
			for(int y=0 ; y<scale_lf.row ; y++){
				for(int x=0 ; x<scale_lf.col ; x++){
					Point4D p(u,v,y,x);
					/*
					if(ref_map!=NULL){
							int y_pos = y/2<ref_map[0][0].row ? y/2 : ref_map[0][0].row-1;
							int x_pos = x/2<ref_map[0][0].col ? x/2 : ref_map[0][0].col-1;
							Point4D shift = (ref_map[u][v].get_value(y_pos, x_pos));
							//if(shift==Point4D(NINF, NINF, NINF, NINF))
							if(shift==Point4D(NINF, NINF, NINF, NINF))
								shift = getRandomPos(lf, Point4D(0,0,0,0), Point4D(0,0,0,0), Point4D(lf.u,lf.v,lf.row,lf.col), NON_MASK);
							else{
								shift.x *= 2; shift.y *= 2;
								if((x&1) || (y&1)){
									Point4D pos = getRandomPos(lf, p+shift, sample_range*(-1), sample_range, NON_MASK);
									if(pos==Point4D(NINF, NINF, NINF, NINF))
										shift = shift;
									else
										shift = shift+pos;
								}
							}
							//std::cout << p+shift <<std::endl;
							//if(lf.in_mask(p+shift))
							//	shift = getRandomPos(lf, Point4D(0,0,0,0), Point4D(0,0,0,0), Point4D(lf.u,lf.v,lf.row,lf.col)) - p;
							set_shift(p, shift, self_map); /// watch out
							//if(dir==COHERENCE)
							//scale_lf.set_value(p, lf.get_value(p+shift, NON_MASK));
							//lf.set_value(p, lf.get_value(p + get_shift(p)));	
							//lf.setDepth(p, lf.getDepth(p + get_shift(p)));
							//min_table->set_value(p, Scalar(patch_diff(lf, p, p+get_shift(p), patch_size)));
							continue;
						}
						*/
						/* else */
						Point4D pos;
						pos = getRandomPos(lf, Point4D(0,0,0,0), Point4D(0,0,0,0), Point4D(lf.u,lf.v,lf.row,lf.col), NON_MASK);
						if(pos.u==NINF)
							std::cout << "wtf!" <<std::endl;
						//std::cout << "set shift" <<std::endl;
						set_shift(p, pos - p, self_map);
						//if(dir==COHERENCE)
					//		scale_lf.set_value(p, lf.get_value(pos, NON_MASK));
					//	std::cout << "set shift done" <<std::endl;
					//	scale_lf.set_value(p, lf.get_value(pos, NON_MASK));
						//scale_lf.setDepth(p, lf.getDepth(pos, NON_MASK));
					}	
				}
			}
	//imshow("tmp", scale_lf.cvimg[0][0]);
	//waitKey(0);
	for(int u=0 ; u<scale_lf.u ; u++)
			for(int v=0 ; v<scale_lf.v ; v++){
				for(int y=0 ; y<scale_lf.row ; y++){
					//std::cout << u << ", " << v << ", " << y << std::endl;
					for(int x=0 ; x<scale_lf.col ; x++){
						Point4D p(u, v, y, x);
						//std::cout << u << ", " << v << ", " << y << ", " << x << "/// "<< lf.row << ", " << std::endl;
						min_table->set_value(p, Scalar(patch_diff(scale_lf, lf, p, p+get_shift(p, self_map), patch_size)));
					}
				}
			}
}


/* PatchMatch::propagation  */
inline bool compare(int index, int bound, int phase)
{
	return phase==0 ? index<=bound : index>=bound;
}
inline double weightKernel(double x)
{
	//return  x==0? 10000 : 1.0/x;
	return 1;
}

void LightFieldRetarget::propagate_and_randomSearch(LightField &lf, LightField &scale_lf, LightFieldData<float, 1> *&min_table, Map **self_map, int phase,
													LightFieldData<float, 4> &weight_table, LightFieldData<float, 1> &depth_table)
{
	Point4D size(lf.u, lf.v, lf.row, lf.col);
	Point4D neighbor_shift[4];
	int even_order_u[] = {0, 0, -1, 0};
	int even_order_v[] = {0, 0, 0, -1};
	int even_order_y[] = {-1, 0, 0, 0};
	int even_order_x[] = {0, -1, 0, 0};
	int start[4], end[4], step[4];
	if(!phase){
		for(int i=0 ; i<4 ; i++){
			neighbor_shift[i].set_value(even_order_u[i], even_order_v[i], even_order_y[i], even_order_x[i]);
			start[i]=0; step[i]=1;
		}
		end[0]=scale_lf.u-1;
		end[1]=scale_lf.v-1;
		end[2]=scale_lf.row-1;
		end[3]=scale_lf.col-1;
	}else{
		for(int i=0 ; i<4 ; i++){
			neighbor_shift[i].set_value(-even_order_u[i], -even_order_v[i], -even_order_y[i], -even_order_x[i]);
			end[i]=0; step[i]=-1;
		}
		start[0]=scale_lf.u-1;
		start[1]=scale_lf.v-1;
		start[2]=scale_lf.row-1;
		start[3]=scale_lf.col-1;
	}
//	LightFieldData<float, 4> weight_table(Point4D(lf.u, lf.u, lf.row, lf.col));
//	LightFieldData<float, 1> depth_table(Point4D(lf.u, lf.u, lf.row, lf.col));
	// Begin to propagate
	for(int u=start[0] ; compare(u,end[0],phase) ; u+=step[0]){
		for(int v=start[1] ; compare(v,end[1],phase) ; v+=step[1]){
			for(int y=start[2] ; compare(y,end[2],phase) ; y+=step[2]){
			//	std::cout<< "propagate: "<<u << ", "<< v << ", " << y<<std::endl;
				for(int x=start[3] ; compare(x,end[3],phase) ; x+=step[3]){
					Point4D p(u,v,y,x);
					Point4D best_shift = get_shift(p, self_map);
					double min = min_table->get_value(p)[0];
					/************** propagation part **************/
					for(int n=0 ; n<4 ; n++){
						Point4D neighbor = p + neighbor_shift[n];
						if(lf.check_4D_bound(neighbor)){
							Point4D neighbor_shift = get_shift(neighbor, self_map);
							Point4D p2 = p + neighbor_shift; //get the shift of neighbor , and shift it.
							if((!lf.check_4D_bound(p2)))
								continue;
							double diff = patch_diff(scale_lf, lf, p, p2, patch_size);
							if(min > diff){
								min = diff;
								best_shift = neighbor_shift;
							}
						}
					}
					//set_shift(p, best_shift, self_map);
					//min_table->set_value(p, Scalar(min));
					//scale_lf.set_value(p, lf.get_value(p + best_shift));
					//lf.setDepth(p, lf.getDepth(p + best_shift));
					//setWeightTable(lf, p, p+get_shift(p), min==0? 10000 : 1.0/min, patch_size, weight_table, depth_table);
			//???		setWeightTable(lf, p, p+best_shift, 1, patch_size, weight_table, depth_table);
					/*********** end of the propagation ***********/
					/*********** start of the random search ***********/
					//double min = min_table->get_value(p)[0];
					//Point4D best_shift = get_shift(p);
					
					Point4D shift = p + best_shift;  // shift = p+shift
					Point4D range = size;
					for(int i=0 ; range.y>1 && range.x>1 ; i++){
						int u_a = shift.u-range.u>=0 ? -range.u : -shift.u;
						int u_b = shift.u+range.u<lf.u ? range.u : lf.u-shift.u;
						int v_a = shift.v-range.v>=0 ? -range.v : -shift.v;
						int v_b = shift.v+range.v<lf.v ? range.v : lf.v-shift.v;
						int y_a = shift.y-range.y>=0 ? -range.y : -shift.y;
						int y_b = shift.y+range.y<lf.row ? range.y : lf.row-shift.y;
						int x_a = shift.x-range.x>=0 ? -range.x : -shift.x;
						int x_b = shift.x+range.x<lf.col ? range.x : lf.col-shift.x;
						Point4D pos = getRandomPos(lf, shift, Point4D(u_a, v_a, y_a, x_a), Point4D(u_b, v_b, y_b, x_b), NON_MASK);
						range = range*0.5;
						if(pos==Point4D(NINF,NINF,NINF,NINF)){
							//std::cout << "fuck" <<std::endl;
							continue;
						}
						Point4D tmp = shift+pos;
						if(!lf.check_4D_bound(tmp)){
							continue;
						}
						double diff = patch_diff(scale_lf, lf, p, tmp, patch_size);
						if(min> diff){
							min = diff;
							best_shift = best_shift+pos;
						}
						
						//if(phase==1)
						//std::cout<< range <<std::endl;
						//std::cout << p << range <<std::endl;
					}
					
					/*********** end of the random search ***********/
					if(dir==COHERENCE)
						setWeightTable(scale_lf, lf, p, p+best_shift, weightKernel(min), patch_size, weight_table, depth_table);  //min==0? 10000 : 1.0/min
					else
						setWeightTable(lf, scale_lf, p+best_shift, p, weightKernel(min), patch_size, weight_table, depth_table);
						
					set_shift(p, best_shift, self_map);
					min_table->set_value(p, Scalar(min));
					//std::cout << "point 3" <<std::endl;
				}
			//char filename[20];
			//sprintf(filename, "iteration1/%d_%d.jpg", u, v);
			//imwrite(filename, lf.img[u][v]);
			}
		}
	}
	/*
	for(int u=0 ; u<scale_lf.u ; u++){
		for(int v=0 ; v<scale_lf.v ; v++){
			for(int y=0 ; y<scale_lf.row ; y++){
				for(int x=0 ; x<scale_lf.col ; x++){
					Point4D p(u, v, y, x);
					double min = min_table->get_value(p)[0];
					Point4D best_shift = get_shift(p, self_map);
					Point4D shift = p + best_shift;  // shift = p+shift
					Point4D range = size;
					for(int i=0 ; range.y>1 && range.x>1 ; i++){
						int u_a = shift.u-range.u>=0 ? -range.u : -shift.u;
						int u_b = shift.u+range.u<lf.u ? range.u : lf.u-shift.u;
						int v_a = shift.v-range.v>=0 ? -range.v : -shift.v;
						int v_b = shift.v+range.v<lf.v ? range.v : lf.v-shift.v;
						int y_a = shift.y-range.y>=0 ? -range.y : -shift.y;
						int y_b = shift.y+range.y<lf.row ? range.y : lf.row-shift.y;
						int x_a = shift.x-range.x>=0 ? -range.x : -shift.x;
						int x_b = shift.x+range.x<lf.col ? range.x : lf.col-shift.x;
						Point4D pos = getRandomPos(lf, shift, Point4D(u_a, v_a, y_a, x_a), Point4D(u_b, v_b, y_b, x_b), NON_MASK);
						range = range*0.5;
						if(pos==Point4D(NINF,NINF,NINF,NINF)){
							//std::cout << "fuck" <<std::endl;
							continue;
						}
						Point4D tmp = shift+pos;
						if(!lf.check_4D_bound(tmp)){
							continue;
						}
						double diff = patch_diff(scale_lf, lf, p, tmp, patch_size);
						if(min> diff){
							min = diff;
							best_shift = best_shift+pos;
						}
						
						//if(phase==1)
						//std::cout<< range <<std::endl;
						//std::cout << p << range <<std::endl;
					}
					set_shift(p, best_shift, self_map);
					min_table->set_value(p, Scalar(min));
					if(dir==COHERENCE)
						setWeightTable(scale_lf, lf, p, p+best_shift, 1, patch_size, weight_table, depth_table);
					else
						setWeightTable(lf, scale_lf, p+best_shift, p, 1, patch_size, weight_table, depth_table);
				}
			}
		}
	}
	*/
	//copyWeightTable(lf, weight_table, depth_table);
}

void LightFieldRetarget::refine(LightField &scale_lf, LightFieldData<float, 4> &weight_table, LightFieldData<float, 1> &depth_table,
								LightFieldData<float, 4> &weight_table_com, LightFieldData<float, 1> &depth_table_com, float scaling_rate)
{
	for(int u=0 ; u<scale_lf.u ; u++)
		for(int v=0 ; v<scale_lf.v ; v++)
			for(int y=0 ; y<scale_lf.row ; y++)
				for(int x=0 ; x<scale_lf.col ; x++){
					Point4D p(u, v, y, x);
					Scalar color_coh = weight_table.get_value(p);
					Scalar color_com = weight_table_com.get_value(p);
					double m = color_coh[3];
					double n = color_com[3];
					Scalar color;
					float coh = n*scaling_rate + m;
					float com = n + m/scaling_rate;
					for(int ch=0 ; ch<3 ; ch++){
						color[ch] = color_coh[ch]/coh + color_com[ch]/com;
						//if(color[ch]<=0)
							//std::cout << color[ch] <<std::endl;
					}
					//if(normalize<=0.0)
						//std::cout << normalize << std::endl;
					scale_lf.set_value(p, color);
					//uchar depth = color[4]/normalize;
					scale_lf.setDepth(p, depth_table.get_value(p)[0]/coh + depth_table_com.get_value(p)[0]/com);
				}
}
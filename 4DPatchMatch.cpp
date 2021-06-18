#include "4DPatchMatch.h"
#include "omp.h"
#include <windows.h>

//float max_d;
//using namespace cv;

std::ostream &operator<<(std::ostream &out, const Point4D &p)
{
	out << "(" << p.u << ", " << p.v << ", " << p.y << ", " << p.x << ")";
	return out;
}

/* random number generator */
inline int random(int a, int b)
{
	if(b-a==0)
		return 0;
	int rand_num = rand()%(b-a);
	return rand_num>=0 ? rand_num+a : rand_num+b;
}
inline float random()
{
	return (rand()%100000)/50000.0 - 1.0;
}

void getNewMapAndMinTable(Map **&map, LightFieldData<float, 1> *&min_table, Point4D size)
{
	min_table = new LightFieldData<float, 1>(size);
	map = new Map*[size.u]; // shift map
	for(int i=0 ; i<size.u ; i++){
		map[i] = new Map[size.v];
		for(int j=0 ; j<size.v ; j++){
			map[i][j] = Map(size.y, size.x);
		}
	}
}

void getWeightTable(Point4D size, Mat **&table)
{
	table = new Mat*[size.u];
	for(int i=0 ; i<size.u ; i++){
		table[i] = new Mat[size.v];
		for(int j=0 ; j<size.v ; j++){
			table[i][j] = Mat::zeros(size.y, size.x, CV_64FC4);
		}
	}
	//return table;
}

void deleteWeightTable(Point4D size, LightFieldData<double, 4> &table, LightFieldData<double, 1> &depth_table)
{
	free(table.lf);
	free(depth_table.lf);
	//delete &table;
	//delete &depth_table;
}

void setWeightTable(LightField &lf, Point4D &src, Point4D &tar, 
					const double weight, const Point4D &patch_size, LightFieldData<float, 4> &table, LightFieldData<float, 1> &depth_table)
{
	for(int u=-patch_size.u ; u<=patch_size.u ; u++)
		for(int v=-patch_size.v ; v<=patch_size.v ; v++)
			for(int y=-patch_size.y ; y<=patch_size.y ; y++)
				for(int x=-patch_size.x ; x<=patch_size.x ; x++){
					Point4D shift = Point4D(u,v,y,x);
					Point4D s = src+shift;
					Point4D t = tar+shift;
					if(!(lf.check_4D_bound(s) && lf.check_4D_bound(t)))
						continue;
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

void copyWeightTable(LightField &lf, LightFieldData<float, 4> &table, LightFieldData<float, 1> &depth_table)
{
	for(int u=0 ; u<lf.u ; u++)
		for(int v=0 ; v<lf.v ; v++)
			for(int y=0 ; y<lf.row ; y++)
				for(int x=0 ; x<lf.col ; x++){
					Point4D p(u, v, y, x);
					if(!lf.in_mask(p))
						continue;
					Scalar color = table.get_value(p);
					//std::cout << color << std::endl;
					double normalize = color[3];
					//double normalize = table[p.u][p.v].at<Vec4d>(p.y, p.x)[3];
					for(int ch=0 ; ch<3 ; ch++){
						color[ch] = color[ch]/normalize;
						//if(color[ch]<=0)
							//std::cout << color[ch] <<std::endl;
					}
					//if(normalize<=0.0)
						//std::cout << normalize << std::endl;
					lf.set_value(p, color);
					//uchar depth = color[4]/normalize;
					lf.setDepth(p, depth_table.get_value(p)[0]/normalize);
				}
}

/* used for computing patch difference */
inline float length(Scalar s)
{
	float dis = 0.0;
	for(int i=0 ; i<3 ; i++)
		dis += s[i]*s[i];
	return (dis);
}

double patch_diff(LightField &lf, Point4D p1, 
				Point4D p2, Point4D patch_size, float max_depth)
{
#define LAMBDA 0.3  // depth weight 0.25
#define BETA 0.2  // consistency weight 0.3
#define NOT_IN_MASK_WEIGHT 32
#define IN_MASK_WEIGHT 1

	int count = 0;
	double diff = 0;
	double depth_diff = 0;  // for depth
	for(int u=-patch_size.u ; u<=patch_size.u ; u++)
		for(int v=-patch_size.v ; v<=patch_size.v ; v++)
			for(int y=-patch_size.y ; y<=patch_size.y ; y++)
				for(int x=-patch_size.x ; x<=patch_size.x ; x++){
					if(!lf.check_4D_bound(p1 + Point4D(u,v,y,x)) || !lf.check_4D_bound(p2 + Point4D(u,v,y,x)))
						continue;
					Scalar color1 = lf.get_value(p1 + Point4D(u,v,y,x), NON_MASK);
					Scalar color2 = lf.get_value(p2 + Point4D(u,v,y,x), NON_MASK);
					uchar depth1 = lf.getDepth(p1 + Point4D(u,v,y,x), NON_MASK); // for depth
					uchar depth2 = lf.getDepth(p2 + Point4D(u,v,y,x), NON_MASK); // for depth
					//int w1=1, w2 = 1;
					int w1 = lf.in_mask(p1 + Point4D(u,v,y,x))? IN_MASK_WEIGHT : NOT_IN_MASK_WEIGHT;
					int w2 = lf.in_mask(p2 + Point4D(u,v,y,x))? IN_MASK_WEIGHT : NOT_IN_MASK_WEIGHT;
					double color_diff = 0;
					for(int ch=0 ; ch<3 ; ch++)
						color_diff += (color1[ch]-color2[ch])*(color1[ch]-color2[ch]);
					diff += (color_diff/255/255/3*w1*w2);
					depth_diff += (((double)depth1-(double)depth2)*((double)depth1-(double)depth2)/255/255*w1*w2);
					count += (w1*w2);
				}
	double intensity =  ((1-LAMBDA)*(diff/count) + LAMBDA*depth_diff/count);
	/////////////////////////////////////////////////////////////////////////////
	Scalar color = lf.get_value(p2, NON_MASK);
	double p2_depth = lf.getDepth(p2, NON_MASK);
	double depth = lf.getDepth(p1, NON_MASK); // for depth
	double disparity = depth/255*2*max_depth - max_depth;
	double error = 0;
	double depth_error = 0;
	for(int v=0 ; v<lf.v ; v++){ //h
		float position = p1.x + (p1.v - v)*disparity;
		int int_position = floor(position);
		Scalar color1 = lf.get_value(Point4D(p1.u, v, p1.y, int_position), NON_MASK);
		Scalar color2 = lf.get_value(Point4D(p1.u, v, p1.y, int_position+1), NON_MASK);
		if(color1[0]<0 || color2[0]<0)
			continue;
		Scalar interpolated_color = (color2*(position-int_position) + color1*(int_position+1-position));
		error += (length(color - interpolated_color)/255/255/3);
		// depth consistency
		double depth1 = lf.getDepth(Point4D(p1.u, v, p1.y, int_position), NON_MASK);
		double depth2 = lf.getDepth(Point4D(p1.u, v, p1.y, int_position+1), NON_MASK);
		double interpolated_depth = (depth2*(position-int_position) + depth1*(int_position+1-position));
		depth_error += ((interpolated_depth - p2_depth)*(interpolated_depth - p2_depth)/255/255);
	}
	for(int u=0 ; u<lf.u ; u++){ //v
		float position = p1.y + (p1.u - u)*(-disparity);
		int int_position = floor(position);
		Scalar color1 = lf.get_value(Point4D(u, p1.v, int_position, p1.x), NON_MASK);
		Scalar color2 = lf.get_value(Point4D(u, p1.v, int_position+1, p1.x), NON_MASK);
		if(color1[0]<0 || color2[0]<0)
			continue;
		Scalar interpolated_color = (color2*(position-int_position) + color1*(int_position+1-position));
		error += (length(color - interpolated_color)/255/255/3);
		// depth consistency
		double depth1 = lf.getDepth(Point4D(u, p1.v, int_position, p1.x), NON_MASK);
		double depth2 = lf.getDepth(Point4D(u, p1.v, int_position+1, p1.x), NON_MASK);
		double interpolated_depth = (depth2*(position-int_position) + depth1*(int_position+1-position));
		depth_error += ((interpolated_depth - p2_depth)*(interpolated_depth - p2_depth)/255/255);
	}
	//double consistency = (1-LAMBDA)*error/(lf.u+lf.v-2) + LAMBDA*depth_error/(lf.u+lf.v-2);
	double consistency = 0.5*error/(lf.u+lf.v-2) + 0.5*depth_error/(lf.u+lf.v-2);
	/////////////////////////////////////////////////////////////////////////////
	return (1-BETA)*intensity + BETA*consistency;
}

/* define PatchMatch::initial */
Point4D getRandomPos(LightField &lf, Point4D base, Point4D min, Point4D max)
{
	Scalar tmp_color(-1, -1, -1);
	Point4D pos;
	int count = 0;
	//std::cout << min << max << std::endl;
	while(tmp_color[0]<0 && count<100){
		pos.u = random(min.u, max.u);
		pos.v = random(min.v, max.v);
		pos.y = random(min.y, max.y);
		pos.x = random(min.x, max.x);
		//std::cout << "pos = "<< pos << std::endl;
		tmp_color = lf.get_value(base+pos);
		count++;
	}
	
	if(count==100)
		return Point4D(NINF, NINF, NINF, NINF);
	return pos;
}

Point4D getRandomPos(LightField &lf, Point4D base, Point4D min, Point4D max, MASK_FLAG mask)
{
	Scalar tmp_color(-1, -1, -1);
	Point4D pos;
	int count = 0;
	//std::cout << min << max << std::endl;
	while(tmp_color[0]<0 && count<100){
		pos.u = random(min.u, max.u);
		pos.v = random(min.v, max.v);
		pos.y = random(min.y, max.y);
		pos.x = random(min.x, max.x);
		//std::cout << "pos = "<< pos << std::endl;
		tmp_color = lf.get_value(base+pos, mask);
		count++;
	}
	
	if(count==100)
		return Point4D(NINF, NINF, NINF, NINF);
	return pos;
}


void PatchMatch::initial(LightField &lf, LightFieldData<float, 1> *&min_table, Map **ref_map, float max_d)
{
//#pragma omp parallel for
		std::cout << "begin to initial " <<std::endl;
		Point4D sample_range(2, 2, lf.row*0.1, lf.col*0.1);
		for(int u=0 ; u<lf.u ; u++)
			for(int v=0 ; v<lf.v ; v++){
				for(int y=0 ; y<lf.row ; y++){
					for(int x=0 ; x<lf.col ; x++){
						Point4D p(u,v,y,x);
						if(!lf.in_mask(p))
							continue;
						if(ref_map!=NULL){
							//std::cout << "haha" <<std::endl;
							int y_pos = y/2<ref_map[0][0].row ? y/2 : ref_map[0][0].row-1;
							int x_pos = x/2<ref_map[0][0].col ? x/2 : ref_map[0][0].col-1;
							Point4D shift = (ref_map[u][v].get_value(y_pos, x_pos));
							//if(shift==Point4D(NINF, NINF, NINF, NINF))
							shift.x *= 2; shift.y *= 2;
							if((x&1) || (y&1)){
								Point4D pos = getRandomPos(lf, p+shift, sample_range*(-1), sample_range);
								if(pos==Point4D(NINF, NINF, NINF, NINF))
									shift = shift;
								else
									shift = shift+pos;
							}
							//std::cout << p+shift <<std::endl;
							if(lf.in_mask(p+shift))
								shift = getRandomPos(lf, Point4D(0,0,0,0), Point4D(0,0,0,0), Point4D(lf.u,lf.v,lf.row,lf.col)) - p;
							set_shift(p, shift); /// watch out
							lf.set_value(p, lf.get_value(p + get_shift(p)));	
							lf.setDepth(p, lf.getDepth(p + get_shift(p)));
							//min_table->set_value(p, Scalar(patch_diff(lf, p, p+get_shift(p), patch_size, max_d)));
							continue;
						}
						//std::cout << "else" <<std::endl;
						/* else */
						Point4D pos;
						pos = getRandomPos(lf, Point4D(0,0,0,0), Point4D(0,0,0,0), Point4D(lf.u,lf.v,lf.row,lf.col));
						set_shift(p, pos - p);
						
						// avoid to race condition
						/*
						while(get_shift(p)==Point4D(0,0,0,0)){  
							set_shift(p, pos-p);
							std::cout << p << std::endl;
						}*/
						//std::cout << p<<", "<<lf.row << "..........." <<std::endl;
						//std::cout << lf.get_value(pos) <<std::endl;
						lf.set_value(p, lf.get_value(pos));
						lf.setDepth(p, lf.getDepth(pos));
						//double tmp = patch_diff(lf, p, p+get_shift(p), patch_size);
						//std::cout << p << "ya" << ", " << Scalar(tmp) <<std::endl;
						//min_table->set_value(p, Scalar(patch_diff(lf, p, p+get_shift(p), patch_size)));
						//std::cout << p  <<std::endl;
						//std::cout << p <<std::endl;
						//min_table[u][v].at<double>(y, x) = patch_diff(lf, p, p+get_shift(p), patch_size);
					}	
				}
				//char filename[20];
				//sprintf(filename, "iteration1/%d_%d.jpg", u, v);
				//imwrite(filename, lf.img[u][v]);
			}
	for(int u=0 ; u<lf.u ; u++)
			for(int v=0 ; v<lf.v ; v++){
				for(int y=0 ; y<lf.row ; y++){
					for(int x=0 ; x<lf.col ; x++){
						Point4D p(u, v, y, x);
						if(!lf.in_mask(p))
							continue;
						min_table->set_value(p, Scalar(patch_diff(lf, p, p+get_shift(p), patch_size, max_d)));
					}
				}
			}
}

/* PatchMatch::propagation  */
inline bool compare(int index, int bound, int phase)
{
	return phase==0 ? index<=bound : index>=bound;
}

void PatchMatch::propagate(LightField &lf, LightFieldData<float, 1> *&min_table, int phase, float max_d)
{
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
		end[0]=lf.u-1;
		end[1]=lf.v-1;
		end[2]=lf.row-1;
		end[3]=lf.col-1;
	}else{
		for(int i=0 ; i<4 ; i++){
			neighbor_shift[i].set_value(-even_order_u[i], -even_order_v[i], -even_order_y[i], -even_order_x[i]);
			end[i]=0; step[i]=-1;
		}
		start[0]=lf.u-1;
		start[1]=lf.v-1;
		start[2]=lf.row-1;
		start[3]=lf.col-1;
	}
	LightFieldData<float, 4> weight_table(Point4D(lf.u, lf.u, lf.row, lf.col));
	LightFieldData<float, 1> depth_table(Point4D(lf.u, lf.u, lf.row, lf.col));
	//getWeightTable(Point4D(lf.u, lf.v, lf.row, lf.col), weight_table);
	// Begin to propagate
	for(int u=start[0] ; compare(u,end[0],phase) ; u+=step[0]){
		for(int v=start[1] ; compare(v,end[1],phase) ; v+=step[1]){
			std::cout<< "propagate: "<<u << ", "<< v <<std::endl;
			for(int y=start[2] ; compare(y,end[2],phase) ; y+=step[2])
				for(int x=start[3] ; compare(x,end[3],phase) ; x+=step[3]){
					Point4D p(u,v,y,x);
					if(!lf.in_mask(p))
						continue;
					Point4D best_shift = get_shift(p);
					double min = min_table->get_value(p)[0];
					/************** core part **************/
					for(int n=0 ; n<4 ; n++){
						Point4D neighbor = p + neighbor_shift[n];
						if(lf.check_4D_bound(neighbor)){
							Point4D neighbor_shift = get_shift(neighbor);
							Point4D p2 = p + neighbor_shift; //get the shift of neighbor , and shift it.
							if((!lf.check_4D_bound(p2)) || lf.in_mask(p2) || p==p2)
								continue;
							double diff = patch_diff(lf, p, p2, patch_size, max_d);
							if(min > diff){
								min = diff;
								best_shift = neighbor_shift;
							}
						}
					}
					set_shift(p, best_shift);
					lf.set_value(p, lf.get_value(p + best_shift));
					lf.setDepth(p, lf.getDepth(p + best_shift));
					min_table->set_value(p, Scalar(min));
					setWeightTable(lf, p, p+get_shift(p), min==0? 10000 : 1.0/min, patch_size, weight_table, depth_table);
					//setWeightTable(lf, p, p+get_shift(p), 1, patch_size, weight_table, depth_table);
					/*********** end of the core ***********/
				}
			//char filename[20];
			//sprintf(filename, "iteration1/%d_%d.jpg", u, v);
			//imwrite(filename, lf.img[u][v]);
		}
	}
	copyWeightTable(lf, weight_table, depth_table);
}

void PatchMatch::random_search(LightField &lf, LightFieldData<float, 1> *&min_table, Point4D size, float max_d)
{
//#pragma omp parallel for
	//Mat **weight_table;
	//getWeightTable(Point4D(lf.u, lf.v, lf.row, lf.col), weight_table);

	//int cannotfind = 0;
	float alpha = 0.5;
	LightFieldData<float, 4> weight_table(Point4D(lf.u, lf.u, lf.row, lf.col));
	LightFieldData<float, 1> depth_table(Point4D(lf.u, lf.u, lf.row, lf.col));
	//Mat **weight_table;
	//getWeightTable(Point4D(lf.u, lf.v, lf.row, lf.col), weight_table);
//#pragma omp parallel for 
	for(int u=0 ; u<lf.u ; u++){
		for(int v=0 ; v<lf.v ; v++){
			std::cout<< "random search: "<< u << ", "<< v <<std::endl;
			for(int y=0 ; y<lf.row ; y++){
				for(int x=0 ; x<lf.col ; x++){
					Point4D p(u,v,y,x);
					if(!lf.in_mask(p))
						continue;
					double min = min_table->get_value(p)[0];
					Point4D best_shift = get_shift(p);
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
						Point4D pos = getRandomPos(lf, shift, Point4D(u_a, v_a, y_a, x_a), Point4D(u_b, v_b, y_b, x_b));
						if(pos==Point4D(NINF,NINF,NINF,NINF))
							continue;
						Point4D tmp = shift+pos;
						if(!lf.check_4D_bound(tmp)){
							continue;
						}
						if(lf.in_mask(tmp))
							continue;
						double diff = patch_diff(lf, p, tmp, patch_size, max_d);
						if(min> diff){
							min = diff;
							best_shift = get_shift(p)+pos;
						}
						range = range*alpha;
						//std::cout << p << range <<std::endl;
					}
					set_shift(p, best_shift);
					lf.set_value(p, lf.get_value(p + best_shift));
					lf.setDepth(p, lf.getDepth(p + best_shift));
					min_table->set_value(p, Scalar(min));
					setWeightTable(lf, p, p+get_shift(p), min==0? 10000 : 1.0/min, patch_size, weight_table, depth_table);
					//setWeightTable(lf, p, p+get_shift(p), 1, patch_size, weight_table, depth_table);
				}
			}
			//char filename[20];
			//sprintf(filename, "iteration1/%d_%d.jpg", u, v);
			//imwrite(filename, lf.img[u][v]);
		}
	}
	copyWeightTable(lf, weight_table, depth_table);
	//deleteWeightTable(Point4D(lf.u, lf.v, lf.row, lf.col), weight_table, depth_table);
	//std::cout << "cannotfind = " << cannotfind << std::endl;
}
/*
uchar findMedian(LightField &lf, Point4D p, int r)
{
	int median[256]={0};
	int num=0;
	for(int u=-r ; u<=r ; u++)
		for(int v=-r ; v<=r ; v++)
	//int u=0 , v=0;
			for(int y=-r ; y<=r ; y++)
				for(int x=-r ; x<=r ; x++){
					if(!lf.check_4D_bound(p+Point4D(u,v,y,x)))
						continue;
					uchar depth = lf.getDepth(p+Point4D(u,v,y,x), NON_MASK);
					median[depth]++;
					num++;
				}
	int count = 0;
	uchar ret = 0;
	for(int i=0 ; i<256 ; i++){
		count += median[i];
		if(count>=(num/2)){
		//	std::cout << num << ", " << i << std::endl;
			ret=i;
			break;
		}
	}
	
	return ret;
}

void PatchMatch::depthMedianBlur(LightField &lf)
{
	LightFieldData<uchar, 1> median(Point4D(lf.u, lf.v, lf.row, lf.col));
	for(int u=0 ; u<lf.u ; u++)
		for(int v=0 ; v<lf.v ; v++)
			for(int y=0 ; y<lf.row ; y++)
				for(int x=0 ; x<lf.col ; x++){
					Point4D p(u, v, y, x);
					if(!lf.in_mask(p))
						continue;
					median.set_value(p, Scalar(findMedian(lf, p, 1)));
						//median.setDepth(p, findMedian(lf, p, 1));
				}
	for(int u=0 ; u<lf.u ; u++)
		for(int v=0 ; v<lf.v ; v++)
			for(int y=0 ; y<lf.row ; y++)
				for(int x=0 ; x<lf.col ; x++){
					Point4D p(u, v, y, x);
					if(lf.in_mask(p))
						lf.setDepth(p, median.get_value(p)[0]);
				}
	std::cout << "median filter done." << std::endl;	
			
}*/
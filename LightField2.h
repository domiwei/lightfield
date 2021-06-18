//#include "LightField.h"
#ifndef __LIGHTFIELDDATA_H__
#define __LIGHTFIELDDATA_H__

using namespace cv;
template<class T, int vector_size>
class LightFieldData{
public:
	LightFieldData(Point4D light_field_size)
	{
		size = light_field_size;
		lf = new T[size.u*size.v*size.y*size.x*sizeof(T)*vector_size];
		stride_y = size.x;
		stride_v = size.x*size.y;
		stride_u = size.x*size.y*size.v;
		for(int i=0 ; i<size.u*size.v*size.y*size.x*sizeof(T)*vector_size ; i++)
			lf[i]=0;

	}
	LightFieldData(Mat **img, int size_u, int size_v)
	{
		size = Point4D(size_u, size_v, img[0][0].rows, img[0][0].cols);
		lf = new T[size.u*size.v*size.y*size.x*sizeof(T)*vector_size];
		stride_y = size.x;
		stride_v = size.x*size.y;
		stride_u = size.x*size.y*size.v;
		for(int u=0 ; u<size_u ; u++){
			for(int v=0 ; v<size_v ; v++){
				for(int y=0 ; y<size.y ; y++){
					for(int x=0 ; x<size.x ; x++){
						if(vector_size == 3){
							Scalar color = img[u][v].at<Vec3b>(y, x);
							lf[vector_size*sizeof(T)*(x + y*stride_y + v*stride_v + u*stride_u)] = color[0];
							lf[vector_size*sizeof(T)*(x + y*stride_y + v*stride_v + u*stride_u) + sizeof(T)] = color[1];
							lf[vector_size*sizeof(T)*(x + y*stride_y + v*stride_v + u*stride_u) + 2*sizeof(T)] = color[2];
						}
						else
							lf[vector_size*sizeof(T)*(x + y*stride_y + v*stride_v + u*stride_u)] = img[u][v].at<uchar>(y, x);
					}
				}
			}
		}
	}
	
	~LightFieldData()
	{
		free(lf);
		std::cout << "--------------free mem----------------" <<std::endl;
	}
	/*
	int lfSize()
	{
		return size.u*size.v*size.y*size.x*sizeof(T)*vector_size;
	}
	*/
	Scalar get_value(Point4D p)
	{
		if(!check_4D_bound(p))	
			return Scalar(-1,-1,-1);
		Scalar ret;
		T ptr[vector_size];
		memcpy(ptr, &lf[vector_size*sizeof(T)*(p.x + p.y*stride_y + p.v*stride_v + p.u*stride_u)], sizeof(T)*vector_size);
		for(int ch=0 ; ch<vector_size ; ch++){
			ret[ch] = ptr[ch];
			//std::cout << p.y << p.x <<"---" <<ret[ch] << ", " << ch << ", " << vector_size << std::endl;
		}
		return ret;
	}
	void set_value(Point4D p, Scalar color)
	{
		T ptr[vector_size];
		for(int ch=0 ; ch<vector_size ; ch++)
			ptr[ch] = color[ch];
		memcpy(&lf[vector_size*sizeof(T)*(p.x + p.y*stride_y + p.v*stride_v + p.u*stride_u)], ptr, sizeof(T)*vector_size);
	}
	bool check_4D_bound(Point4D p)
	{
		return (check_bound(p.u, p.v, size.u, size.v) && check_bound(p.y, p.x, size.y, size.x));
	}
	Point4D size;
	T *lf;
private:
	
	int stride_y, stride_v, stride_u;
	
};

#endif
#include "LightField.h"
#include <opencv2\imgproc\imgproc.hpp>
#ifndef __PYRAMID_H__
#define __PYRAMID_H__
class LightFieldPyramid{
	public:
		int layer;
		LightFieldPyramid(int layer){
			this->layer = layer;
		}
		void pyramidInitial(LightField lf);
		LightField *getLayer(int l){ 
			if(l>=layer) return NULL; 
			else return pyramid[l];
		}
		void setLayer(LightField &lf, int layer);
		~LightFieldPyramid();
	private:
		vector<LightField *> pyramid;
};

#endif
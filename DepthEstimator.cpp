#include "DepthEstimator.h"


inline float length(Scalar s)
{
	float dis = 0.0;
	for(int i=0 ; i<3 ; i++)
		dis += s[i]*s[i];
	return (dis);
}

void DepthEstimator::confidenceEstimator(LightFieldData<uchar, 3> &lf, Point4D p, float &conx, float &cony)
{
	float threshold = 0.02;
	// u, y is fixed.
	Scalar center = lf.get_value(p)/255;
	conx=0.0;
	for(int dx=-4 ; dx<=4 ; dx++){
		Point4D p2 = p+Point4D(0,0,0,dx);
		Scalar color = lf.get_value(p2)/255;
		if(color[0]<0) continue;
		conx += length(color-center);
	}
	cony=0.0;
	// v, x is fixed.
	for(int dy=-4 ; dy<=4 ; dy++){
		Point4D p2 = p+Point4D(0,0,dy,0);
		Scalar color = lf.get_value(p2)/255;
		if(color[0]<0) continue;
		cony += length(color-center);
	}
}


float kernel(float dis)
{
	float ratio = dis/0.02;
	//return (1-ratio*ratio);
	return ratio<=1?(1-ratio*ratio):0;
}
float DepthEstimator::depthEstimator(LightFieldData<uchar, 3> &lf, Point4D p, Trust trust)
{
	float step = MAX_DEPTH*2.0/DEPTH_SAMPLE;
	Scalar center = lf.get_value(p);
	// u, y is fixed.
	float depth=0.0;
	float max = -1000;
	int size = (trust==TRUST_X?lf.size.v:lf.size.u);
	for(float shift=-MAX_DEPTH ; shift<MAX_DEPTH ; shift+=step){
		int count = 0;
		float error = 0.0;
		for(int v=0 ; v<size ; v++){
			float disparity = (trust==TRUST_X?p.x+(p.v - v)*shift:p.y+(p.u - v)*shift);
			int int_disparity = floor(disparity);
			Scalar color1, color2;
			///
			if(trust == TRUST_X){
				color1 = lf.get_value(Point4D(p.u, v, p.y, int_disparity));
				color2 = lf.get_value(Point4D(p.u, v, p.y, int_disparity+1));
			}else{
				color1 = lf.get_value(Point4D(v, p.v, int_disparity, p.x));
				color2 = lf.get_value(Point4D(v, p.v, int_disparity+1, p.x));
			}
			///
			if(color1[0]<0 || color2[0]<0)
				continue;
			Scalar color = (color2*(disparity-int_disparity) + color1*(int_disparity+1-disparity));
			float score = kernel(length(color-center)/(255*255));
			error += score;
			count++;
		}
		error /= count;
		//std::cout <<error<<std::endl;
		if(max<error){
			max = error;
			depth = shift;
		}
	}
	//if(depth==0.0)
	//std::cout <<depth<<std::endl;
	return (trust==TRUST_X?depth:-depth);;
}

void DepthEstimator::depthPropagate(Point4D p, float depth, LightFieldData<uchar, 3> &lf)
{
	for(int v=0 ; v<lf.size.v ; v++){ //h
		float position = p.x + (p.v - v)*depth;
	//	position = (position-(int)position<0.2 || (int)position+1-position>0.8) ? (int)position : (int)position+1;
		if(position-(int)position<=0.2 || position-(int)position>=0.8)
			position = (int)position;
		else
			continue;
			
		if(!lf.check_4D_bound(Point4D(p.u,v,p.y,position)))
			continue;
		if(depth_map[p.u][v].at<float>(p.y, position)!=0.0)
			continue;
		depth_map[p.u][v].at<float>(p.y, position) = depth;
	}
	
	for(int u=0 ; u<lf.size.u ; u++){ //v
		float position = p.y + (p.u - u)*(-depth);
	//	position = position-(int)position<0.5? (int)position : (int)position+1;	
		if(position-(int)position<=0.2 || position-(int)position>=0.8)
			position = (int)position;
		else
			continue;
			
		if(!lf.check_4D_bound(Point4D(u,p.v,position,p.x)))
			continue;
		if(depth_map[u][p.v].at<float>(position, p.x)!=0.0)
			continue;
		depth_map[u][p.v].at<float>(position, p.x) = depth;
	}
	
}





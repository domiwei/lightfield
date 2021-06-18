#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "LightFieldPyramid.h"
#include <iostream>     /* srand, rand */
#include <time.h>       /* time */
#define MAX_DEPTH 2

//#include "4DPoint.h"
//using namespace cv;
void getNewMapAndMinTable(Map **&map, LightFieldData<float, 1> *&min_table, Point4D size);

class PatchMatch{
public:
	PatchMatch(){}
	PatchMatch(LightField &lf)
	{
		srand (time(NULL));
		this->_lf = lf;
		map = NULL;
	};
	void run_4d_patchmatch(LightField &lf, int size_u, int size_v, 
							int size_y, int size_x, int layer = 1)
	{
		clock_t start_time, end_time;
		patch_size.set_value(size_u, size_v, size_y, size_x);
		LightFieldPyramid lfp(layer);  
		std::cout << "begin to generate pyramid" <<std::endl;
		lfp.pyramidInitial(lf); // set pyramid
		std::cout << "generating pyramid done!" <<std::endl;
		
		map = NULL;
		for(int l=layer-1 ; l>=0 ; l--){
			float max_d = MAX_DEPTH / pow(2.0, l);
			LightField *now_lf = lfp.getLayer(l);
			Map **ref_map = map;
			LightFieldData<float, 1> *min_table;
			getNewMapAndMinTable(map, min_table, Point4D(now_lf->u, now_lf->v, now_lf->row, now_lf->col));
			initial(*now_lf, min_table, ref_map, max_d);
			std::cout << "initial done" <<std::endl;
			Point4D range(now_lf->u, now_lf->v, now_lf->row, now_lf->col);
			for(int i=0 ; i<l/(1.5)+2 ; i++){
				if(l>=4)
					patch_size = Point4D(size_u, size_v, 2, 2);
				else
					patch_size = Point4D(size_u, size_v, size_y, size_x);
				propagate(*now_lf, min_table, i%2, max_d);  // propagation
				//random_search(*now_lf, min_table, range, max_d);  // random search
				//depthMedianBlur(*now_lf);
			}
			std::cout << "Layer " << l << " done." << std::endl;
			char test[20];
			sprintf(test, "out6/%d.jpg", l);
			now_lf->writeImage(test, 0, 0);
			// free the memory //
			delete min_table;

		}
		
		/*	for(int u=0 ; u<lf.u ; u++)
			for(int v=0 ; v<lf.v ; v++)
				for(int y=0 ; y<lf.row ; y++)
					for(int x=0 ; x<lf.col ; x++){
						Point4D p(u,v,y,x);
						if(!lf.in_mask(p))
							continue;
						lf.set_value(p, lf.get_value(p + get_shift(p)));
					}*/
	}
private:
	LightField _lf;
	Map **map;
	Point4D patch_size;
	void initial(LightField &lf, LightFieldData<float, 1> *&min_table, Map **ref_map, float max_d);
	void propagate(LightField &lf, LightFieldData<float, 1> *&min_table, int phase, float max_d); //, double (*patch_diff)(LightField&, Point4D, Point4D, Point4D));
	void random_search(LightField &lf, LightFieldData<float, 1> *&min_table, Point4D range, float max_d); //, double (*patch_diff)(LightField&, Point4D, Point4D, Point4D));
	void depthMedianBlur(LightField &lf);
	//void getNewMapAndMinTable(Map **&map, LightFieldData<float, 1> *&min_table, Point4D size);
	Point4D get_shift(Point4D p){ return map[p.u][p.v].get_value(p.y, p.x); }
	void set_shift(Point4D &p, Point4D &shift){ map[p.u][p.v].set_value(p.y, p.x, shift); }
};
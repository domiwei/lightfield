#include <iostream>
#define NINF -10000
//namespace shift{
#ifndef __4DPOINT_H__
#define __4DPOINT_H__
	struct Point4D{
		Point4D();
		Point4D(int su, int sv, int sy, int sx){ set_value(su, sv, sy, sx); }
		Point4D operator+(const Point4D &rhs);
		Point4D operator-(const Point4D &rhs);
		Point4D operator*(const float &mul);
	//	Point4D operator=(const Point4D &rhs);
		bool operator==(const Point4D &rhs);
		//friend std::ostream &operator<<(std::ostream &out, const Point4D &p);
		void set_value(int su, int sv, int sy, int sx);
		int u, v, y, x;
		~Point4D(){
			//std::cout << "delete 4d point........" <<std::endl;
		}
	};

	struct Map{
		Map(){}
		Map(int row, int col);
		void set_value(int y, int x, int su, int sv, int sy, int sx);
		void set_value(int y, int x, Point4D value);
		Point4D get_value(int y, int x);
		Point4D **map;
		int row, col;
		~Map(){
			//std::cout << "delete 4d map........" <<std::endl;
			//for(int i=0 ; i<row ; i++) //
				//for(int j=0 ; j<col ; j++)
					//free(&map[i][j]);
			//free(map);
		}
	};

#endif


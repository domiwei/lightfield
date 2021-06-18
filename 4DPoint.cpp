#include "4DPoint.h"
////////////////// namespace shift ////////////////////

/* Point4D constructor*/
Point4D::Point4D()
{
		u = v = y = x = 0;
}
void Point4D::set_value(int su, int sv, int sy, int sx)
{
	this->u = su;
	this->v = sv;
	this->y = sy;
	this->x = sx;
}
/* Map constructor */
Map::Map(int row, int col){
		map = new Point4D*[row];
		for(int i=0 ; i<row ; i++){
			map[i] = new Point4D[col];
			for(int j=0 ; j<col ; j++)
				map[i][j] = Point4D();
		}
		this->row = row;
		this->col = col;
	}
/* Map::set_value */
void Map::set_value(int y, int x, int su, int sv, int sy, int sx)
	{
		map[y][x].set_value(su, sv, sy, sx);
	}

Point4D Point4D::operator+(const Point4D &rhs)
{
	Point4D p;
	p.set_value(this->u + rhs.u,
				this->v + rhs.v, 
				this->y + rhs.y, 
				this->x + rhs.x);
	return p;
}

Point4D Point4D::operator-(const Point4D &rhs)
{
	Point4D p;
	p.set_value(this->u - rhs.u,
				this->v - rhs.v, 
				this->y - rhs.y, 
				this->x - rhs.x);
	return p;
}

bool Point4D::operator==(const Point4D &rhs)
{
	return this->u==rhs.u && this->v==rhs.v && this->y==rhs.y && this->x==rhs.x;
}

Point4D Point4D::operator*(const float &mul)
{
	return Point4D(this->u*mul, this->v*mul, this->y*mul, this->x*mul);
}
/*
Point4D Point4D::operator=(const Point4D &rhs)
{
	return Point4D(rhs.u, rhs.v, rhs.y, rhs.x);
}
*/

Point4D Map::get_value(int y, int x)
{
	if(y<0 || y>=row || x<0 || x>=col)
		return Point4D(NINF, NINF, NINF, NINF);
	return map[y][x];
}

void Map::set_value(int y, int x, Point4D v)
{
	map[y][x].set_value(v.u, v.v, v.y, v.x);
}


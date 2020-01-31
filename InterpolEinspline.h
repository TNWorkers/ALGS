#ifndef INTERPOLEINSPLINE
#define INTERPOLEINSPLINE

#include "InterpolBase.h"
#include <armadillo>
#include <einspline/bspline.h> // Einspline

template<>
class Interpol<EINSPLINE> : public virtual InterpolBase<EINSPLINE>
{
public:
	
	Interpol (double xmin_input, double xmax_input, int xPoints_input);
	Interpol (initializer_list<double> rmin_input, initializer_list<double> rmax_input, initializer_list<int> points_input);
	
	void set_splines();
	void kill_splines();
	
//	void insert (int ix, int iy, int iz, double value);
	
	// with range check:
	double evaluate (double x);
	double evaluate (double x, double y);
	double evaluate (double x, double y, double z);
	
	// no range check:
	double operator() (double x);
	double operator() (double x, double y);
	double operator() (double x, double y, double z);
	
private:
	
	void resize();
	
	// data:
	arma::cube data3d;
	
	// Einspline:
	UBspline_1d_d * spline1d;
	UBspline_2d_d * spline2d;
	UBspline_3d_d * spline3d;
};

//---------------<constructors & destructors>---------------------
Interpol<EINSPLINE>::
Interpol (double xmin_input, double xmax_input, int xPoints_input)
:InterpolBase<EINSPLINE> (xmin_input, xmax_input, xPoints_input)
{
	resize();
}

Interpol<EINSPLINE>::
Interpol (initializer_list<double> rmin_input, initializer_list<double> rmax_input, initializer_list<int> points_input)
:InterpolBase<EINSPLINE> (rmin_input, rmax_input, points_input)
{
	resize();
}

void Interpol<EINSPLINE>::
resize()
{
	if      (dim==1) {data1d.resize(points[0]);}
	else if (dim==2) {data2d.resize(points[0],points[1]);}
	else if (dim==3) {data3d.resize(points[0],points[1],points[2]);}
}
//---------------</constructors & destructors>---------------------

//---------------<inserting>---------------------
//inline void Interpol<EINSPLINE>::
//insert (int ix, int iy, int iz, double value)
//{
//	data3d(ix,iy,iz) = value;
//}
//---------------</inserting>---------------------

//---------------<set and kill splines>---------------------
void Interpol<EINSPLINE>::
set_splines()
{
	BCtype_d xBC = {NATURAL, NATURAL, 0., 0.};
	
	vector<Ugrid> TensorGrid;
	for (int i=0; i<dim; ++i)
	{
		Ugrid Grid;
		Grid.start = rmin[i];
		Grid.end   = rmax[i];
		Grid.num   = points[i];
		TensorGrid.push_back(Grid);
	}
	
	if      (dim==1) {spline1d = create_UBspline_1d_d(TensorGrid[0], xBC, data1d.data());}
	else if (dim==2) {spline2d = create_UBspline_2d_d(TensorGrid[0], TensorGrid[1], xBC, xBC, data2d.data());}
	else if (dim==3) {spline3d = create_UBspline_3d_d(TensorGrid[2], TensorGrid[1], TensorGrid[0], xBC, xBC, xBC, data3d.memptr());}

	SPLINES_ARE_SET = true;
}

void Interpol<EINSPLINE>::
kill_splines()
{
	if      (dim==1) {destroy_Bspline(spline1d);}
	else if (dim==2) {destroy_Bspline(spline2d);}
	else if (dim==3) {destroy_Bspline(spline3d);}
}
//---------------</set and kill splines>---------------------

//---------------<evaluate 1dim>---------------------
double Interpol<EINSPLINE>::
evaluate (double x)
{
	assert(dim==1);
	if (SPLINES_ARE_SET==false) {set_splines();}
	double out;
	eval_UBspline_1d_d(spline1d, x, &out);
	return out;
}

double Interpol<EINSPLINE>::
operator() (double x)
{
	double out;
	eval_UBspline_1d_d(spline1d, x, &out);
	return out;
}
//---------------</evaluate 1dim>---------------------

//---------------<evaluate 2dim>---------------------
double Interpol<EINSPLINE>::
evaluate (double x, double y)
{
	assert(dim==2);
	if (SPLINES_ARE_SET==false) {set_splines();}
	double out;
	eval_UBspline_2d_d(spline2d, x, y, &out);
	return out;
}

double Interpol<EINSPLINE>::
operator() (double x, double y)
{
	double out;
	eval_UBspline_2d_d(spline2d, x, y, &out);
	return out;
}
//---------------</evaluate 2dim>---------------------

//---------------<evaluate 3dim>---------------------
double Interpol<EINSPLINE>::
evaluate (double x, double y, double z)
{
	assert(dim==3);
	if (SPLINES_ARE_SET==false) {set_splines();}
	double out;
	eval_UBspline_3d_d(spline3d, z, y, x, &out);
	return out;
}

double Interpol<EINSPLINE>::
operator() (double x, double y, double z)
{
	double out;
	eval_UBspline_3d_d(spline3d, z, y, x, &out);
	return out;
}
//---------------</evaluate 3dim>---------------------

#endif

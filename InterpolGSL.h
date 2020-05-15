#ifndef INTERPOL1DIM
#define INTERPOL1DIM

#include "InterpolBase.h"
#include <gsl/gsl_spline.h> // GSL Akima
// requires GSL 2.0:
#include <gsl/gsl_interp2d.h> 
#include <gsl/gsl_spline2d.h>

//enum GSL_SPLINE_TYPE {LINEAR, POLYNOMIAL, CSPLINE, CSPLINE_PERIODIC, AKIMA, AKIMA_PERIODIC};

template<>
class Interpol<GSL> : public virtual InterpolBase<GSL>
{
public:
	
	Interpol(){};
	Interpol (double xmin_input, double xmax_input, int xPoints_input);
	Interpol (const Eigen::VectorXd &x_axis);
	Interpol (const vector<double>  &x_axis);
	Interpol (const vector<pair<double,double> > &xy);
	Interpol (initializer_list<double> rmin_input, initializer_list<double> rmax_input, initializer_list<int> points_input);
	
	void set_splines();
	void kill_splines();
	
	// with checks:
	double evaluate (double x);
	double deriv (double x);
	double deriv2 (double x);
	double integral (double xmin, double xmax);
	double evaluate (double x, double y);
	
	double deriv_x  (double x, double y);
	double deriv_y  (double x, double y);
	
	// without checks:
	double operator() (double x) const;
	double operator() (double x, double y) const;
	
	template<typename MatrixType> void operator= (const MatrixType &M);
	void operator= (const Eigen::VectorXd &V);
	void operator= (const vector<double>  &V);
	
	double integrate();
	void setConstant (double c);
	
private:
	
	Eigen::VectorXd data_x;
	
	gsl_spline   * spline1d;
	gsl_spline2d * spline2d;
	
	gsl_interp_accel * xbooster;
	gsl_interp_accel * ybooster;
};

//---------------<constructors & destructors>---------------------
Interpol<GSL>::
Interpol (double xmin_input, double xmax_input, int xPoints_input)
:InterpolBase<GSL> (xmin_input, xmax_input, xPoints_input)
{
	resize();
}

Interpol<GSL>::
Interpol (const Eigen::VectorXd &x_axis)
:InterpolBase<GSL> (x_axis(0), x_axis(x_axis.rows()-1), x_axis.rows())
{
	resize();
	data_x = x_axis;
}

Interpol<GSL>::
Interpol (const vector<double> &x_axis)
:InterpolBase<GSL> (x_axis[0], x_axis[x_axis.size()-1], x_axis.size())
{
	resize();
	data_x.resize(x_axis.size());
	for (int i=0; i<x_axis.size(); ++i)
	{
		data_x(i) = x_axis[i];
	}
}

Interpol<GSL>::
Interpol (const vector<pair<double,double> > &xy_input)
:InterpolBase<GSL> (xy_input[0].first, xy_input[xy_input.size()-1].first, xy_input.size())
{
	resize();
	
	vector<pair<double,double> > xy = xy_input;
	xy.erase(unique(xy.begin(),xy.end()), xy.end());
	sort(xy.begin(), xy.end());
	
	data_x.resize(xy.size());
	data1d.resize(xy.size());
	
	for (int i=0; i<xy.size(); ++i)
	{
		data_x(i) = xy[i].first;
		data1d(i) = xy[i].second;
	}
}

Interpol<GSL>::
Interpol (initializer_list<double> rmin_input, initializer_list<double> rmax_input, initializer_list<int> points_input)
:InterpolBase<GSL> (rmin_input, rmax_input, points_input)
{
	assert(rmin_input.size() <= 2 and rmax_input.size() <= 2 and points_input.size() <= 2);
	resize();
}
//---------------</constructors & destructors>---------------------

//---------------<set and kill splines>---------------------
void Interpol<GSL>::
set_splines()
{
//	Eigen::VectorXd x_axis(points[0]);
//	for (int ix=0; ix<points[0]; ++ix) {x_axis(ix) = rmin[0] + (rmax[0]-rmin[0])*ix/(points[0]-1);}
	
	if (dim == 1)
	{
		Eigen::VectorXd x_axis(points[0]);
		if (data_x.rows() == 0)
		{
			for (int ix=0; ix<points[0]; ++ix) {x_axis(ix) = rmin[0] + (rmax[0]-rmin[0])*ix/(points[0]-1);}
		}
		else
		{
			x_axis = data_x;
		}
		
		xbooster = gsl_interp_accel_alloc();
		spline1d = gsl_spline_alloc(gsl_interp_akima, points[0]);
		gsl_spline_init(spline1d, x_axis.data(), data1d.data(), points[0]);
	}
	else if (dim == 2)
	{
//		double xa[] = {rmin[0], rmax[0]};
//		double ya[] = {rmin[1], rmax[1]};
//		cout << rmin[0] << "\t" << rmax[0] << endl;
//		cout << rmin[1] << "\t" << rmax[1] << endl;
//		cout << points[0] << "\t" << points[1] << endl;
		
		data2d.transposeInPlace(); // GSL requires column-major storage
		
		Eigen::VectorXd x_axis(points[0]);
		for (int ix=0; ix<points[0]; ++ix) {x_axis(ix) = rmin[0] + (rmax[0]-rmin[0])*ix/(points[0]-1);}
		
		Eigen::VectorXd y_axis(points[1]);
		for (int iy=0; iy<points[1]; ++iy) {y_axis(iy) = rmin[1] + (rmax[1]-rmin[1])*iy/(points[1]-1);}
		
		xbooster = gsl_interp_accel_alloc();
		ybooster = gsl_interp_accel_alloc();
		spline2d = gsl_spline2d_alloc(gsl_interp2d_bicubic, points[0], points[1]);
		gsl_spline2d_init(spline2d, x_axis.data(), y_axis.data(), data2d.data(), points[0], points[1]);
	}
	
	this->SPLINES_ARE_SET = true;
}

void Interpol<GSL>::
kill_splines()
{
	if (dim == 1)
	{
		gsl_spline_free(spline1d);
		gsl_interp_accel_free(xbooster);
	}
	else if (dim == 2)
	{
		gsl_spline2d_free(spline2d);
		gsl_interp_accel_free(xbooster);
		gsl_interp_accel_free(ybooster);
	}
}
//---------------</set and kill splines>---------------------

//---------------<evaluate 1dim>---------------------
double Interpol<GSL>::
evaluate (double x)
{
	assert(dim == 1);
	if (SPLINES_ARE_SET == false) {set_splines();}
	return gsl_spline_eval(spline1d, x, xbooster);
}

double Interpol<GSL>::
operator() (double x) const
{
	return gsl_spline_eval(spline1d, x, xbooster);
}

double Interpol<GSL>::
deriv (double x)
{
	assert(dim == 1);
	if (SPLINES_ARE_SET == false) {set_splines();}
	return gsl_spline_eval_deriv(spline1d, x, xbooster);
}

double Interpol<GSL>::
deriv2 (double x)
{
	assert(dim == 1);
	if (SPLINES_ARE_SET == false) {set_splines();}
	return gsl_spline_eval_deriv2(spline1d, x, xbooster);
}

double Interpol<GSL>::
deriv_x (double x, double y)
{
	assert(dim == 2);
	if (SPLINES_ARE_SET == false) {set_splines();}
	return gsl_spline2d_eval_deriv_x(spline2d, x, y, xbooster, ybooster);
}

double Interpol<GSL>::
deriv_y (double x, double y)
{
	assert(dim == 2);
	if (SPLINES_ARE_SET == false) {set_splines();}
	return gsl_spline2d_eval_deriv_y(spline2d, x, y, xbooster, ybooster);
}

double Interpol<GSL>::
integral (double xmin, double xmax)
{
	assert(dim == 1);
	if (SPLINES_ARE_SET == false) {set_splines();}
	return gsl_spline_eval_integ(spline1d, xmin, xmax, xbooster);
}
//---------------</evaluate 1dim>---------------------

//---------------<evaluate 2dim>---------------------
double Interpol<GSL>::
evaluate (double x, double y)
{
	assert(dim == 2);
	if (SPLINES_ARE_SET == false) {set_splines();}
	return gsl_spline2d_eval(spline2d, x, y, xbooster, ybooster);
}

double Interpol<GSL>::
operator() (double x, double y) const
{
	return gsl_spline2d_eval(spline2d, x, y, xbooster, ybooster);
}
//---------------</evaluate 1dim>---------------------

//---------------<equate with vector>---------------------
template<typename MatrixType>
void Interpol<GSL>::
operator= (const MatrixType &M)
{
	assert(dim == 1 and points[0] == M.rows());
//	for (int ix=0; ix<points[0]; ++ix)
//	{
//		insert(ix,M(ix));
//	}
	assert(M.cols() == 2);
	data_x = M.col(0);
	data1d.resize(data_x.rows());
	for (int ix=0; ix<M.rows(); ++ix)
	{
		insert(ix,M(ix,1));
	}
}

void Interpol<GSL>::
operator= (const Eigen::VectorXd &V)
{
	data1d = V;
}

void Interpol<GSL>::
operator= (const vector<double> &V)
{
	data1d.resize(V.size());
	for (int i=0; i<V.size(); ++i)
	{
		data1d(i) = V[i];
	}
}
//---------------</equate with vector>---------------------

double Interpol<GSL>::
integrate()
{
	if (SPLINES_ARE_SET == false) {set_splines();}
	return gsl_spline_eval_integ(spline1d, rmin[0], rmax[0], xbooster);
}

inline void Interpol<GSL>::
setConstant (double c)
{
	data1d.setConstant(c);
}

#endif

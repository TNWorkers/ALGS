#ifndef INTERPOLBASE
#define INTERPOLBASE

using namespace std;

#include <Eigen/Dense>
//#include <armadillo>

enum SPLINE_LIBRARY {GSL, EINSPLINE};

template<SPLINE_LIBRARY SLIB>
class InterpolBase
{
public:
	
	InterpolBase(){};
	InterpolBase (double xmin_input, double xmax_input, int xPoints_input);
	InterpolBase (initializer_list<double> rmin_input, initializer_list<double> rmax_input, initializer_list<int> points_input);
	~InterpolBase();
	
	void insert (int ix, double value);
	void insert (int ix, int iy, double value);
	
	virtual void kill_splines(){};
	
	inline vector<double> get_rmin() {return rmin;}
	inline vector<double> get_rmax() {return rmax;}
	
	inline Eigen::VectorXd get_data1d() const {return data1d;};
	
protected:
	
	virtual void resize();
	
	int dim;
	
	// data:
	Eigen::VectorXd data1d;
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> data2d;
	
	vector<double> rmin;
	vector<double> rmax;
	vector<int>    points;
	
	bool SPLINES_ARE_SET;
};

//---------------<specialized>---------------------
template<SPLINE_LIBRARY SLIB>
class Interpol : protected InterpolBase<SLIB>
{};
//---------------</specialized>---------------------

//---------------<constructors>---------------------
template<SPLINE_LIBRARY SLIB>
InterpolBase<SLIB>::
InterpolBase (double xmin_input, double xmax_input, int xPoints_input)
:dim(1), SPLINES_ARE_SET(false)
{
	rmin.push_back(xmin_input);
	rmax.push_back(xmax_input);
	points.push_back(xPoints_input);
	data1d.resize(points[0]);
}

template<SPLINE_LIBRARY SLIB>
InterpolBase<SLIB>::
InterpolBase (initializer_list<double> rmin_input, initializer_list<double> rmax_input, initializer_list<int> points_input)
:SPLINES_ARE_SET(false)
{
	dim = rmin_input.size();
	assert(rmin_input.size() == rmax_input.size() and 
	       rmin_input.size() == points_input.size() and 
	       rmax_input.size() == points_input.size() and 
	       dim <= 3);
	
	for (auto it=rmin_input.begin(); it!=rmin_input.end(); ++it) {rmin.push_back(*it);}
	for (auto it=rmax_input.begin(); it!=rmax_input.end(); ++it) {rmax.push_back(*it);}
	for (auto it=points_input.begin(); it!=points_input.end(); ++it) {points.push_back(*it);}
}

template<SPLINE_LIBRARY SLIB>
void InterpolBase<SLIB>::
resize()
{
	if      (dim==1) {data1d.resize(points[0]);}
	else if (dim==2) {data2d.resize(points[0],points[1]);}
}

template<SPLINE_LIBRARY SLIB>
InterpolBase<SLIB>::
~InterpolBase()
{
	if (SPLINES_ARE_SET == true) {kill_splines();}
}
//---------------</constructors>---------------------

//---------------<insertion>---------------------
template<SPLINE_LIBRARY SLIB>
inline void InterpolBase<SLIB>::
insert (int ix, double value)
{
	data1d(ix) = value;
}

template<SPLINE_LIBRARY SLIB>
inline void InterpolBase<SLIB>::
insert (int ix, int iy, double value)
{
	data2d(ix,iy) = value;
}
//---------------</insertion>---------------------

#endif

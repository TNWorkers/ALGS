#ifndef COMPLEXINTERPOLGSL
#define COMPLEXINTERPOLGSL

#include "InterpolGSL.h"

struct ComplexInterpol
{
	ComplexInterpol(){};
	
	ComplexInterpol (Eigen::ArrayXd axis)
	{
		for (int c=0; c<2; ++c) data[c] = Interpol<GSL>(axis);
	}
	
	void insert (int i, complex<double> val)
	{
		data[0].insert(i, val.real());
		data[1].insert(i, val.imag());
	}
	
	complex<double> integrate()
	{
		return data[0].integrate() + 1.i * data[1].integrate();
	};
	
	complex<double> evaluate (double x)
	{
		return data[0].evaluate(x) + 1.i * data[1].evaluate(x);
	};
	
	std::array<Interpol<GSL>,2> data; // array indices: real and imaginary part
	
	void kill_splines()
	{
		data[0].kill_splines();
		data[1].kill_splines();
	}
};

#endif

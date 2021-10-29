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
	};
	
	void operator= (const VectorXcd &V)
	{
		for (int i=0; i<V.rows(); ++i)
		{
			data[0].insert(i, V(i).real());
			data[1].insert(i, V(i).imag());
		}
	};
	
	complex<double> integrate()
	{
		return data[0].integrate() + 1.i * data[1].integrate();
	};
	
	complex<double> evaluate (double x)
	{
		return data[0].evaluate(x) + 1.i * data[1].evaluate(x);
	};
	
	double evaluateRe (double x)
	{
		return data[0].evaluate(x);
	};
	
	double evaluateIm (double x)
	{
		return data[1].evaluate(x);
	};
	
	double quick_evaluateRe (double x) const
	{
		return data[0](x);
	};
	
	double quick_evaluateIm (double x) const
	{
		return data[1](x);
	};
	
	complex<double> operator() (double x) const
	{
		return data[0](x) + 1.i * data[1](x);
	}
	
	std::array<Interpol<GSL>,2> data; // array indices: real and imaginary part
	
	void set_splines()
	{
		data[0].set_splines();
		data[1].set_splines();
	}
	
	void kill_splines()
	{
		data[0].kill_splines();
		data[1].kill_splines();
	}
};

#endif

#ifndef ORTHPOLYEVAL
#define ORTHPOLYEVAL

#include <Eigen/Dense>
#include <complex>

//#include <gsl/gsl_sf_bessel.h>
#include <boost/math/special_functions.hpp>

double orthpoly_eval (int n, double x, const Eigen::ArrayXd &alpha, const Eigen::ArrayXd &beta)
{
	if (n==0) {return 1.;}
	
	double pm1 = 1.;
	double pm2 = 0.;
	double p = pm1;
	
	int N = alpha.rows()-1;
	assert(N>=n);
	Eigen::ArrayXd A = alpha.head(N).array() / beta.segment(1,N).array().sqrt();
	Eigen::ArrayXd B = beta.segment(0,N).array().sqrt() / beta.segment(1,N).array().sqrt();
	Eigen::ArrayXd C = 1. / beta.segment(1,N).array().sqrt();
	
	for (int k=1; k<=n; ++k)
	{
		p = (C(k-1)*x-A(k-1))*pm1 - B(k-1)*pm2;
		pm2 = pm1;
		pm1 = p;
	}
	return p;
}

double orthpoly_eval (int n, double x, const Eigen::ArrayXd &A, const Eigen::ArrayXd &B, const Eigen::ArrayXd &C)
{
	assert(A.rows()-1 >= n);
	
	if (n==0) {return 1.;}
	
	double pm1 = 1.;
	double pm2 = 0.;
	double p = pm1;
	
	for (int k=1; k<=n; ++k)
	{
		p = (C(k-1)*x-A(k-1))*pm1 - B(k-1)*pm2;
		pm2 = pm1;
		pm1 = p;
	}
	return p;
}

// calculate Chebyshev expansion using Clenshaw recurrence
double ChebyshevExpansion (double x, const Eigen::VectorXd &c)
{
	int N = c.rows();
	double y2 = 0.;
	double y1 = 0.;
//	VectorXd y(N+2);
//	y(N+1) = 0.;
//	y(N) = 0.;
//	for (int i=N-1; i>=0; --i)
//	{
//		double fac = (i>0)? 2. : 1.;
//		y(i) = fac*c(i) + 2.*x*y(i+1) - y(i+2);
//	}
//	return y(0)-x*y(1);
	for (int i=N-1; i>=0; --i)
	{
		double fac = (i>0)? 2. : 1.;
		double y0 = fac*c(i) + 2.*x*y1 - y2;
		y2 = y1;
		y1 = y0;
	}
	return y1-x*y2;
}

enum ORTHPOLY {CHEBYSHEV, LEGENDRE};

std::ostream& operator<< (std::ostream& s, ORTHPOLY P)
{
	if      (P==CHEBYSHEV) {s << "Chebyshev";}
	else if (P==LEGENDRE)  {s << "Legendre";}
	return s;
}

template<ORTHPOLY P>
struct OrthPoly
{
	static double A (int n);
	static double B (int n);
	static double C (int n);
	static double orthfac (int n);
	static double w (double x);
	static double eval (int n, double x);
	static double eval (int n, double x, const Eigen::ArrayXd alpha, const Eigen::ArrayXd beta);
	static double eval (int n, double x, const Eigen::ArrayXd A, const Eigen::ArrayXd B, const Eigen::ArrayXd C);
	static complex<double> evalFT (int, double t);
};

template<ORTHPOLY P>
inline double OrthPoly<P>::
A (int n)
{
	return 0.;
}

template<ORTHPOLY P>
double OrthPoly<P>::
eval (int n, double x)
{
	if      (n==0) {return 1.;}
	else if (n==1) {return x;}
	
	double pm1 = x;
	double pm2 = 1.;
	double p = pm1;
	
	for (int k=2; k<=n; ++k)
	{
		p = (C(k)*x-A(k))*pm1 - B(k)*pm2;
		pm2 = pm1;
		pm1 = p;
	}
	return p;
}

//---- Chebyshev ----
template<>
inline double OrthPoly<CHEBYSHEV>::
C (int n)
{
	return 2.;
}

template<>
inline double OrthPoly<CHEBYSHEV>::
B (int n)
{
	return 1.;
}

template<>
inline double OrthPoly<CHEBYSHEV>::
orthfac (int n)
{
	return (n==0)? 1. : 2.;
}

template<>
inline double OrthPoly<CHEBYSHEV>::
w (double x)
{
	return pow(M_PI*sqrt(1.-x*x),-1.);
}

template<>
inline complex<double> OrthPoly<CHEBYSHEV>::
evalFT (int n, double t)
{
	return pow(-1.i,n) * boost::math::cyl_bessel_j(n,t);
}

//---- Legendre ----
template<>
inline double OrthPoly<LEGENDRE>::
C (int n)
{
	return (2.*n-1.)/n;
}

template<>
inline double OrthPoly<LEGENDRE>::
B (int n)
{
	return (n-1.)/n;
}

template<>
inline double OrthPoly<LEGENDRE>::
orthfac (int n)
{
	return n+0.5;
}

template<>
inline double OrthPoly<LEGENDRE>::
w (double x)
{
	return 1.;
}

template<>
inline complex<double> OrthPoly<LEGENDRE>::
evalFT (int n, double t)
{
	return pow(-1.i,n) * sqrt(2.*M_PI/t) * boost::math::cyl_bessel_j(n+0.5,t);
}

#endif

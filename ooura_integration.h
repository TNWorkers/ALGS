#ifndef MY_OOURA_INTEGRATION
#define MY_OOURA_INTEGRATION

#include <math.h>
#include <complex.h>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

#include "IntervalIterator.h"
#include "ComplexInterpolGSL.h"

/**Source: Tukuya Ooura (Kyoto University ♥), "A Double Exponential Formula for the Fourier Transforms" (2005)*/
namespace Ooura
{
	/**variable transformation*/
	double phi (const int &n, const double &h, const double &a, const double &b)
	{
		double t = n*h;
		
		// careful limit t->0
		return (n==0)? 1./(a+b+2.):
		               t/(1.-exp(-2.*t-a*(1.-exp(-t))-b*(exp(t)-1.)));
	}
	
	/**derivative of phi*/
	double phi_prime (const int &n, const double &h, const double &a, const double &b)
	{
		double t = n*h;
		
		double denom = (1.-exp(-2.*t-a*(1.-exp(-t))-b*(exp(t)-1.)));
		double double_exp = exp(-2.*t-a*(1.-exp(-t))-b*(exp(t)-1.));
		double double_exp_deriv = -2.-a*exp(-t)-b*exp(t);
		
		return (n==0)? 0.5*(a*a+2.*a*b+5.*a+b*b+3.*b+4.)/pow(a+b+2.,2):
		               t*double_exp*double_exp_deriv/pow(denom,2)+1./denom;
	}
	
	/**phi-t*/
	double phi_hat (const int &n, const double &h, const double &a, const double &b)
	{
		return phi(n,h,a,b)-n*h;
	}
	
	/**how to choose the a-parameter of phi*/
	double choose_a (const double &w0, const double &h, const double& b)
	{
		return b/sqrt(1.+log(1.+M_PI/(w0*h))/(4.*w0*h));
	}
	
	/**Find the summation limits Nm and Np so that the Ooura abscissa runs between 0 and tmax*/
	std::pair<int,int> find_Npm (const double &tmax, const double &w0, const double &h, const double &a, const double &b, const double &tmin=1e-9)
	{
		int Nm = 1;
		int Np = 1;
		
		double t_test = M_PI/(w0*h) * phi(0,h,a,b);
		while (t_test>tmin)
		{
			t_test = M_PI/(w0*h) * phi(-(++Nm),h,a,b);
		}
		
		t_test = M_PI/(w0*h) * phi(0,h,a,b);
		while (t_test<tmax)
		{
			t_test = M_PI/(w0*h) * phi(++Np,h,a,b);
		}
		if (t_test>tmax and Np>1) --Np;
		return make_pair(Nm,Np);
	}
	
	/**Fourier transform*/
	VectorXcd FT (const VectorXd &tvals, const VectorXcd &fvals, double h, double wmin, double wmax, int wpoints)
	{
//		cout << "tvals=" << tvals.transpose() << endl;
//		cout << "fvals=" << fvals.transpose() << endl;
//		cout << "h=" << h << endl;
//		cout << "wmin=" << wmin << ", wmax=" << wmax << ", wpoints=" << wpoints << endl;
		
		double tmax = tvals(tvals.rows()-1);
		double w0 = 0.5*wmax;
		double b = 0.25;
		double a = choose_a(w0,h,b);
		std::pair<int,int> Npm = find_Npm(tmax,w0,h,a,b);
		int failsafe = 0;
		while (Npm.second==1 and failsafe<1000)
		{
			w0 += 0.5*wmax;
			Npm = find_Npm(tmax,w0,h,a,b);
			++failsafe;
			if (failsafe==1000) lout << termcolor::red << "Warning: bad OOURA integration!" << termcolor::reset << endl;
		}
		int Nm = Npm.first;
		int Np = Npm.second;
		int Nooura = Np+Nm+1;
		
		// precalc Ooura quantities that are not dependent on omega
		VectorXd ooura_abscissa1(Nooura);
		VectorXd ooura_abscissa2(Nooura);
		VectorXd ooura_weights(Nooura);
		for (auto [i,n] = std::tuple{0,-Nm}; i<Nooura; ++i, ++n)
		{
			ooura_abscissa1(i) = M_PI/(w0*h) * phi(n,h,a,b);
			ooura_abscissa2(i) = M_PI/(2.*h) * phi_hat(n,h,a,b);
			ooura_weights(i)   = sin(ooura_abscissa2(i)) * phi_prime(n,h,a,b);
		}
		
		// setup Interpol
		ComplexInterpol fInterp(tvals);
		for (int i=0; i<fvals.rows(); ++i)
		{
			fInterp.insert(i,fvals(i));
		}
		fInterp.set_splines();
		
		// interpolate function on Ooura grid
		VectorXcd ooura_fvals(Nooura);
		for (int i=0; i<Nooura; ++i)
		{
			ooura_fvals(i) = fInterp.evaluate(ooura_abscissa1(i));
		}
		complex<double> zeroFreqIntegral = fInterp.integrate();
		fInterp.kill_splines();
		
		VectorXcd res(wpoints);
		
		// integrate function
		IntervalIterator w(wmin,wmax,wpoints);
		for (w=w.begin(); w!=w.end(); ++w)
		{
			double omega = *w;
			complex<double> FT_res = 0;
			
			// if omega==0, use gsl spline integration
			if (omega == 0.)
			{
				FT_res = zeroFreqIntegral;
			}
			else
			{
				for (auto [i,n] = std::tuple{0,-Nm}; i<Nooura; ++i, ++n)
				{
					complex<double> weight = ooura_weights(i) * exp(1.i*(abs(omega)*ooura_abscissa1(i)-ooura_abscissa2(i)));
					if (omega<0.)
					{
						// get negative frequencies by exploiting complex conjugation:
						// [∫dt f*(t) exp(i*w*t)]* = ∫dt f(t) exp(-i*w*t)
						FT_res += conj(ooura_fvals(i)) * weight;
					}
					else
					{
						FT_res += ooura_fvals(i) * weight;
					}
				}
				FT_res *= 2.i*M_PI/w0;
				if (omega<0.) FT_res = conj(FT_res);
			}
			
			res(w.index()) = FT_res;
		}
		
		return res;
	}
}

#endif

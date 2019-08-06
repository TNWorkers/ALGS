#ifndef ROOTFINDER
#define ROOTFINDER

#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <complex.h>
#include <vector>
#include "assert.h"

inline complex<double> DivDiff (complex<double> z0, complex<double> z1, complex<double> (*fC)(complex<double>))
{
	return (fC(z1)-fC(z0))/(z1-z0);
}

inline complex<double> DivDiff (complex<double> z0, complex<double> z1, complex<double> z2, complex<double> (*fC)(complex<double>))
{
	return (DivDiff(z1,z2,fC)-DivDiff(z0,z1,fC))/(z2-z0);
}

struct DivideByRoots
{
	static complex<double> (*fC)(complex<double>);
	static std::vector<complex<double> > known_roots;
	
	static inline complex<double> fCprogress (complex<double> z)
	{
		complex<double> out = fC(z);
		for (int i=0; i<known_roots.size(); ++i)
		{
			out /= (z-known_roots[i]);
		}
		return out;
	}
};
complex<double> (*DivideByRoots::fC)(complex<double>) = NULL;
std::vector<complex<double> > DivideByRoots::known_roots;

enum ROOT_ALGORITHM {BISECTION, FALSEPOS, BRENT, MULLER};

class RootFinder
{
public:
	
	RootFinder (double (*f_input)(double,void*), double xmin_input, double xmax_input);
	RootFinder (double (*f_input)(double,void*), double xmin_input, double xmax_input, ROOT_ALGORITHM RootAlgorithm_input);
	RootFinder (double (*f_input)(double,void*));
	~RootFinder();
	
	inline double root();
	
	inline bool test (double xmin_input, double xmax_input);
	inline double root (double xmin_input, double xmax_input);
	std::vector<double> roots (initializer_list<double> limits);
	
	// complex Muller algorithm
	RootFinder (complex<double> (*fC_input)(complex<double>));
	complex<double> root (complex<double> guess);
	complex<double> Muller (complex<double> guess, complex<double> (*fC_input)(complex<double>));
	std::vector<complex<double> > known_roots;
	std::vector<complex<double> > roots (complex<double> guess, int roots_amount);

private:
	
	gsl_function f;
	ROOT_ALGORITHM RootAlgorithm;
	
	double xmin;
	double xmax;
	double x0;
	
	void solve();
	void newton();
	double root_res;
	
	int status;
	int iter;
	int max_iter;
	
	const gsl_root_fsolver_type * gslRootSolverType;
	gsl_root_fsolver * gslRootSolver;

	// complex	
	complex<double> (*fC)(complex<double>);
	complex<double> rmm;
	complex<double> rm;
	complex<double> r;
	complex<double> r_new;
};

RootFinder::
RootFinder (double (*f_input)(double,void*), double xmin_input, double xmax_input)
:xmin(xmin_input), xmax(xmax_input)
{
	RootAlgorithm = BRENT;
	gslRootSolverType = gsl_root_fsolver_brent;
	gslRootSolver = gsl_root_fsolver_alloc(gslRootSolverType);
	f.function = f_input;
	gsl_root_fsolver_set (gslRootSolver, &f, xmin, xmax);

	max_iter = 500;	
	solve();
}

RootFinder::
RootFinder (double (*f_input)(double,void*))
{
	RootAlgorithm = BRENT;
	gslRootSolverType = gsl_root_fsolver_brent;
	gslRootSolver = gsl_root_fsolver_alloc(gslRootSolverType);
	f.function = f_input;

	max_iter = 500;
}

RootFinder::
RootFinder (double (*f_input)(double,void*), double xmin_input, double xmax_input, ROOT_ALGORITHM RootAlgorithm_input)
:xmin(xmin_input), xmax(xmax_input)
{
	RootAlgorithm = RootAlgorithm_input;
	if (RootAlgorithm==BISECTION) {gslRootSolverType = gsl_root_fsolver_bisection;}
	if (RootAlgorithm==FALSEPOS)  {gslRootSolverType = gsl_root_fsolver_falsepos;}
	if (RootAlgorithm==BRENT)     {gslRootSolverType = gsl_root_fsolver_brent;}
	
	gslRootSolver = gsl_root_fsolver_alloc(gslRootSolverType);
	f.function = f_input;
	gsl_root_fsolver_set (gslRootSolver, &f, xmin, xmax);

	max_iter = 500;
	solve();
}

RootFinder::
RootFinder(complex<double> (*fC_input)(complex<double>))
{
	RootAlgorithm = MULLER;
	fC = fC_input;
	DivideByRoots::fC = fC_input;
}

RootFinder::
~RootFinder()
{
	if (RootAlgorithm==BISECTION || RootAlgorithm==BRENT || RootAlgorithm==FALSEPOS)
	{
		gsl_root_fsolver_free (gslRootSolver);
	}
}

void RootFinder::
solve()
{
	iter = 0;
	root_res = 0.;
	do
	{
		++iter;
		status = gsl_root_fsolver_iterate(gslRootSolver);
		root_res = gsl_root_fsolver_root(gslRootSolver);
		xmin = gsl_root_fsolver_x_lower(gslRootSolver);
		xmax = gsl_root_fsolver_x_upper(gslRootSolver);
		status = gsl_root_test_interval (xmin, xmax, 0, 1e-10);
	}
	while (status == GSL_CONTINUE && iter < max_iter);
}

inline bool RootFinder::
test (double xmin_input, double xmax_input)
{
	if (GSL_FN_EVAL(&f,xmin_input)*GSL_FN_EVAL(&f,xmax_input) < 0.) {return true;}
	else {return false;}
}

inline double RootFinder::
root (double xmin_input, double xmax_input)
{
	xmin = xmin_input;
	xmax = xmax_input;
	gsl_root_fsolver_set (gslRootSolver, &f, xmin, xmax);
	solve();
	return root_res;
}

complex<double> RootFinder::
root (complex<double> guess)
{
	return Muller(guess,fC);
}

std::vector<complex<double> > RootFinder::
roots (complex<double> guess, int roots_amount)
{
	DivideByRoots::known_roots.clear();
	known_roots.clear();
	
	for (int i=0; i<roots_amount; ++i)
	{
		complex<double> new_root = Muller(guess,DivideByRoots::fCprogress);
		known_roots.push_back(new_root);
		DivideByRoots::known_roots.push_back(new_root);
	}
	return known_roots;
}

complex<double> RootFinder::
Muller (complex<double> guess, complex<double> (*fC_input)(complex<double>))
{
	double diff_rel = 1.;
	double diff_abs = 1.;

	complex<double> h = 0.1*complex<double>(-1.+(double)(rand()%1000)/500,-1.+(double)(rand()%1000)/500);
	rmm = guess+h;
	rm  = guess;
	r   = guess-h;
	
	int i=0;
	while (diff_rel>1e-10 && diff_abs>1e-6)
	{
		++i;
		complex<double> w = DivDiff(r,rm,fC_input) + DivDiff(r,rmm,fC_input) - DivDiff(rm,rmm,fC_input);
		
		complex<double> sqroot = sqrt(w*w-4.*DivDiff(r,rm,rmm,fC_input));
		if (norm(w-sqroot) > norm(w+sqroot)) {r_new = r - 2.*fC_input(r)/(w-sqroot);}
		else                                 {r_new = r - 2.*fC_input(r)/(w+sqroot);}
		
		diff_abs = abs(r_new-r);
		diff_rel = fabs(1.-abs(r)/abs(r_new));
		
		complex<double> rm_backup = rm;
		complex<double> r_backup = r;
		
		r = r_new;
		rm = r_backup;
		rmm = rm_backup;
	}
	cout << "iterations: " << i << endl;
	return r;
}

double RootFinder::
root()
{
	return root_res;
}

std::vector<double> RootFinder::
roots (initializer_list<double> limits)
{
	assert(limits.size() % 2 == 0);
	std::vector<double> out;
	const double* it=begin(limits);
	while (it != end(limits))
	{
		double xl=*it;
		++it;
		double xr=*it;
		if (test(xl,xr) == true)
		{
			out.push_back(root(xl,xr));
		}
		++it;
	}
	return out;
}

#endif

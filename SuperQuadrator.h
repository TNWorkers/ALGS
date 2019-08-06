#ifndef SUPERQUADRATOR
#define SUPERQUADRATOR

#include "Quadrator.h"
#include "ChebyshevTransformer.h"

template<QUAD_RULE qrule>
class SuperQuadrator
{
public:
	
	SuperQuadrator(){};
	SuperQuadrator (double (*w)(double), double xmin, double xmax, int xpoints);
	
	double abscissa (int k);
	double weight (int k);
	
	double integrate (double (*f)(double));
	
	Eigen::VectorXd get_weights();
	Eigen::VectorXd get_abscissa();
	Eigen::VectorXd get_steps();
	
private:

	Eigen::VectorXd weights;
	Eigen::VectorXd points;
};

template<QUAD_RULE qrule>
SuperQuadrator<qrule>::
SuperQuadrator (double (*w)(double), double xmin, double xmax, int xpoints)
{
	ChebyshevTransformer Charlie(w,xmin,xmax);
//	if (xpoints<=60)
	if (xpoints<=30)
	{
		Charlie.compute_recursionCoefficients_GanderKarp<qrule>(xpoints);
	}
	else
	{
		Charlie.compute_recursionCoefficients_modChebyshev(xpoints);
	}
	
	// using Eigen
	Eigen::MatrixXd Jacobi(xpoints,xpoints);
	Jacobi.setZero();
	for (int k=0; k<xpoints-1; ++k)
	{
		Jacobi(k,k)   = Charlie.alpha()(k);
		Jacobi(k,k+1) = sqrt(Charlie.beta()(k+1));
		Jacobi(k+1,k) = Jacobi(k,k+1);
	}
	Jacobi(xpoints-1,xpoints-1) =  Charlie.alpha()(xpoints-1);
	
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Eugen(Jacobi);
	points = Eugen.eigenvalues();
	weights = Charlie.beta()(0)*Eugen.eigenvectors().row(0).array().square().matrix();
	
//	// using LAPACK
//	VectorXd Jacobi_diag(xpoints);
//	Jacobi_diag.setZero();
//	VectorXd Jacobi_subdiag(xpoints-1);
//	for (int n=0; n<xpoints-1; ++n)
//	{
//		Jacobi_diag(n) = Charlie.alpha()(n);
//		Jacobi_subdiag(n) = sqrt(Charlie.beta()(n+1));
//	}
//	Jacobi_diag(xpoints-1) =  Charlie.alpha()(xpoints-1);
	
//	// using LAPACK; doing stuff the cool way
//	VectorXd Jacobi_diag = Charlie.alpha().head(xpoints);
//	VectorXd Jacobi_subdiag = Charlie.beta().segment(1,xpoints-1).cwiseSqrt();
//	
//	TridiagonalEigensolver Tim(Jacobi_diag,Jacobi_subdiag);
//	Tim.compute_eigvalzerosq(Jacobi_diag,Jacobi_subdiag);
//	
//	points = Tim.eigenvalues();
//	weights = Charlie.beta()(0)*Tim.eigvalzerosq();
}

template<QUAD_RULE qrule>
inline double SuperQuadrator<qrule>::
abscissa (int k)
{
	return points(k);
}

template<QUAD_RULE qrule>
inline double SuperQuadrator<qrule>::
weight (int k)
{
	return weights(k);
}

template<QUAD_RULE qrule>
inline double SuperQuadrator<qrule>::
integrate (double (*f)(double))
{
	double res = 0.;
	#pragma omp parallel for reduction(+:res)
	for (int ix=0; ix<points.rows(); ++ix)
	{
		res += weights(ix) * f(points(ix));
	}
	return res;
}

template<QUAD_RULE qrule>
inline Eigen::VectorXd SuperQuadrator<qrule>::
get_weights()
{
	return weights;
}

template<QUAD_RULE qrule>
inline Eigen::VectorXd SuperQuadrator<qrule>::
get_abscissa()
{
	return points;
}

template<QUAD_RULE qrule>
inline Eigen::VectorXd SuperQuadrator<qrule>::
get_steps()
{
	Eigen::VectorXd Vout(points.rows());
	Vout(0) = points(0);
	for (int i=1; i<points.rows(); ++i)
	{
		Vout(i) = points(i)-points(i-1);
	}
	return Vout;
}

#endif

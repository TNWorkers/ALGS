#ifndef GENERICArnoldiSolver
#define GENERICArnoldiSolver

#ifndef ARNOLDI_MAX_ITERATIONS
#define ARNOLDI_MAX_ITERATIONS 1e2
#endif

#include "RandomVector.h"

template<typename MatrixType, typename VectorType>
class ArnoldiSolver
{
public:
	
	ArnoldiSolver(){};
	
	ArnoldiSolver (const MatrixType &A, VectorType &x, complex<double> &lambda, double tol=1e-14);
	
	void calc_dominant (const MatrixType &A, VectorType &x, complex<double> &lambda, double tol=1e-14);
	
	void set_dimK (size_t dimK_input);
	
	string info() const;
	
private:
	
	size_t dimK, dimA;
	double error;
	size_t N_iter;
	
	bool USER_HAS_FORCED_DIMK;
	
	vector<VectorType> Kbasis;
	
	void iteration (const MatrixType &A, const VectorType &x0, VectorType &x, complex<double> &lambda);
};

template<typename MatrixType, typename VectorType>
string ArnoldiSolver<MatrixType,VectorType>::
info() const
{
	stringstream ss;
	
	ss << "ArnoldiSolver" << ":"
	<< " dimA=" << dimA
	<< ", dimK=" << dimK
	<< ", iterations=" << N_iter;
	if (N_iter == ARNOLDI_MAX_ITERATIONS)
	{
		ss << ", breakoff after max.iterations";
	}
	ss << ", error=" << error;
	
	return ss.str();
}

template<typename MatrixType, typename VectorType>
ArnoldiSolver<MatrixType,VectorType>::
ArnoldiSolver (const MatrixType &A, VectorType &x, complex<double> &lambda, double tol)
{
	calc_dominant(A,x,lambda,tol);
}

template<typename MatrixType, typename VectorType>
void ArnoldiSolver<MatrixType,VectorType>::
set_dimK (size_t dimK_input)
{
	dimK = dimK_input;
	USER_HAS_FORCED_DIMK = true;
}

template<typename MatrixType, typename VectorType>
void ArnoldiSolver<MatrixType,VectorType>::
calc_dominant (const MatrixType &A, VectorType &x, complex<double> &lambda, double tol)
{
	size_t try_dimA = dim(A);
	size_t try_dimx = dim(x);
	assert(try_dimA != 0 or try_dimx != 0);
	dimA = max(try_dimA, try_dimx);
	
	N_iter = 0;
	
	if (!USER_HAS_FORCED_DIMK)
	{
		if      (dimA==1)             {dimK=1;}
		else if (dimA>1 and dimA<200) {dimK=static_cast<size_t>(ceil(max(2.,0.4*dimA)));}
		else                          {dimK=100;}
	}
	
	VectorType x0 = x;
	GaussianRandomVector<VectorType,complex<double> >::fill(dimA,x0);
	
	complex<double> lambda_old = complex<double>(1e3,1e3);
	
	do
	{
		lambda_old = lambda;
		iteration(A,x0,x,lambda); ++N_iter;
		x0 = x;
		error = abs(lambda-lambda_old);
	}
	while (error>tol and N_iter<ARNOLDI_MAX_ITERATIONS);
}

template<typename MatrixType, typename VectorType>
void ArnoldiSolver<MatrixType,VectorType>::
iteration (const MatrixType &A, const VectorType &x0, VectorType &x, complex<double> &lambda)
{
	Kbasis.clear();
	Kbasis.resize(dimK+1);
	Kbasis[0] = x0;
	normalize(Kbasis[0]);
	
	// overlap matrix
	MatrixXcd h(dimK+1,dimK); h.setZero();
	
	// Arnoldi construction of an orthogonal Krylov space basis
	for (size_t j=0; j<dimK; ++j)
	{
		HxV(A,Kbasis[j], Kbasis[j+1]);
		for (size_t i=0; i<=j; ++i)
		{
			h(i,j) = dot(Kbasis[i],Kbasis[j+1]);
			Kbasis[j+1] -= h(i,j) * Kbasis[i];
		}
		h(j+1,j) = norm(Kbasis[j+1]);
		Kbasis[j+1] /= h(j+1,j);
	}
	
	// calculate dominant eigenvector within the Krylov space
	ComplexEigenSolver<MatrixXcd> Eugen(h.topRows(dimK));
	size_t max;
	Eugen.eigenvalues().cwiseAbs().maxCoeff(&max);
	lambda = Eugen.eigenvalues()(max);
	
	// project out of Krylov space
	x = Eugen.eigenvectors().col(max)(0) * Kbasis[0];
	for (size_t k=1; k<dimK; ++k)
	{
		x += Eugen.eigenvectors().col(max)(k) * Kbasis[k];
	}
}

#endif

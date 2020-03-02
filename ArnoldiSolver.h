#ifndef GENERICArnoldiSolver
#define GENERICArnoldiSolver

#ifndef ARNOLDI_MAX_ITERATIONS
#define ARNOLDI_MAX_ITERATIONS 1e2
#endif

#include <algorithm>
#include "RandomVector.h"

template<typename MatrixType, typename VectorType>
class ArnoldiSolver
{
public:
	
	ArnoldiSolver(){};
	
	ArnoldiSolver (const MatrixType &A, VectorType &x, complex<double> &lambda_res, double tol_input=1e-14);
	
	void calc_dominant (const MatrixType &A, VectorType &x, complex<double> &lambda_res, double tol_input=1e-14);
	
	void set_dimK (size_t dimK_input);
	
	string info() const;
	
	complex<double> get_lambda2() const {return lambda2;};
	
private:
	
	size_t dimA, dimK, dimKc;
	double error = 1.;
	size_t N_iter;
	double tol;
	
	complex<double> lambda;
	complex<double> lambda2;
	
	bool USER_HAS_FORCED_DIMK=false;
	
	vector<VectorType> Kbasis;
	
	void iteration (const MatrixType &A, const VectorType &x0, VectorType &x, complex<double> &lambda_res);
};

template<typename MatrixType, typename VectorType>
string ArnoldiSolver<MatrixType,VectorType>::
info() const
{
	stringstream ss;
	
	ss << "ArnoldiSolver" << ":"
	<< " dimA=" << dimA
	<< ", dimKmax=" << dimK
	<< ", dimK=" << dimKc
	<< ", iterations=" << N_iter;
	if (N_iter == ARNOLDI_MAX_ITERATIONS)
	{
		ss << ", breakoff after max.iterations";
	}
	ss << ", error=" << error;
	ss << ", λ1=" << lambda;
	ss << ", |λ1|=" << abs(lambda);
	ss << ", λ2=" << lambda2;
	ss << ", |λ2|=" << abs(lambda2);
	
	return ss.str();
}

template<typename MatrixType, typename VectorType>
ArnoldiSolver<MatrixType,VectorType>::
ArnoldiSolver (const MatrixType &A, VectorType &x, complex<double> &lambda_res, double tol_input)
{
	calc_dominant(A,x,lambda_res,tol_input);
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
calc_dominant (const MatrixType &A, VectorType &x, complex<double> &lambda_res, double tol_input)
{
	tol = tol_input;
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
	normalize(x0);
	do
	{
		iteration(A,x0,x,lambda); ++N_iter;
		x0 = x;
	}
	while (error>tol and N_iter<ARNOLDI_MAX_ITERATIONS);
	
	lambda_res = lambda;
}

complex<double> find_second_largest (const VectorXcd &v)
{
	vector<tuple<complex<double>,int>> vals;
	for (int i=0; i<v.rows(); ++i) vals.push_back(make_tuple(v(i),i));
	sort(vals.begin(), vals.end(), [](tuple<complex<double>,int> a, tuple<complex<double>,int> b)
	{
		if (abs(get<0>(a)) == abs(get<0>(b)))
			return abs(get<0>(a)) > abs(get<0>(b));
		return abs(get<0>(a)) > abs(get<0>(b));
	});
	// get<1> if index is required
//	cout << "largest=" << get<0>(vals[0]) << endl;
	return get<0>(vals[1]);
}


template<typename MatrixType, typename VectorType>
void ArnoldiSolver<MatrixType,VectorType>::
iteration (const MatrixType &A, const VectorType &x0, VectorType &x, complex<double> &lambda_res)
{
	Kbasis.clear();
	Kbasis.resize(dimK+1);
	Kbasis[0] = x0;
	normalize(Kbasis[0]);
	// overlap matrix
	MatrixXcd h(dimK+1,dimK); h.setZero();
	ComplexEigenSolver<MatrixXcd> Eugen;
	size_t max;
	
	dimKc = 1; // current Krylov dimension
	complex<double> lambda_old = complex<double>(1e3,1e3);
	// Arnoldi construction of an orthogonal Krylov space basis
	for (size_t j=0; j<dimK; ++j)
	{
		HxV(A,Kbasis[j], Kbasis[j+1]);
		for (size_t i=0; i<=j; ++i)
		{
			h(i,j) = dot(Kbasis[i],Kbasis[j+1]);
//			Kbasis[j+1] -= h(i,j) * Kbasis[i];
			addScale(-h(i,j),Kbasis[i], Kbasis[j+1]);
		}
		h(j+1,j) = norm(Kbasis[j+1]);
		Kbasis[j+1] /= h(j+1,j);
		
		dimKc = j+1;
		
		// calculate dominant eigenvector within the Krylov space
		Eugen.compute(h.topLeftCorner(dimKc,dimKc));
		Eugen.eigenvalues().cwiseAbs().maxCoeff(&max);
		
		lambda2 = find_second_largest(Eugen.eigenvalues());
//		cout << "second=" << find_second_largest(Eugen.eigenvalues()) << endl;
		
		lambda = Eugen.eigenvalues()(max);
		
		error = abs(lambda_res-lambda_old);
		lambda_old = lambda;
		
		if (error < tol) {break;}
	}
	
	// project out from Krylov space
	x = Eugen.eigenvectors().col(max)(0) * Kbasis[0];
	assert(dimKc == Eugen.eigenvectors().rows() and "Bad Krylov matrix in ArndoldiSolver! Perhaps nan or all zero.");
	for (size_t k=1; k<dimKc; ++k)
	{
//		x += Eugen.eigenvectors().col(max)(k) * Kbasis[k];
		addScale(Eugen.eigenvectors().col(max)(k),Kbasis[k], x);
	}
}

#endif

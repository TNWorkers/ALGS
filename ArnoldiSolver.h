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
	
	ArnoldiSolver (int nmax, double tol_input=1e-14)
	:tol(tol_input)
	{
		errors.resize(nmax);
		lambda.resize(nmax);
		index.resize(nmax);
		Psi.resize(nmax);
	}
	
	ArnoldiSolver (const MatrixType &A, VectorType &x, int nmax, double tol_input=1e-14);
	
	void calc_dominant (const MatrixType &A, VectorType &x);
	
	void set_dimK (size_t dimK_input);
	
	string info() const;
	
	complex<double> get_lambda  (int n) const {return lambda[n];};
	VectorType get_excited      (int n) const {return Psi[n];};
	
private:
	
	size_t dimA, dimK, dimKc;
	double error = 1.;
	size_t N_iter;
	double tol;
	
	vector<double> errors;
	vector<complex<double> > lambda;
	vector<int> index;
	vector<VectorType> Psi;
	
	bool USER_HAS_FORCED_DIMK=false;
	
	vector<VectorType> Kbasis;
	
	void iteration (const MatrixType &A, const VectorType &x0, VectorType &x);
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
	ss << ", errors=";
	for (int n=0; n<errors.size(); ++n)
	{
		ss << errors[n];
		if (n!=errors.size()-1) ss << ", ";
	}
	ss << endl;
	for (int n=0; n<lambda.size(); ++n)
	{
		ss << "λ" << n << "=" << lambda[n] << ", |λ" << n << "|=" << abs(lambda[n]) << endl;
	}
	
	return ss.str();
}

template<typename MatrixType, typename VectorType>
ArnoldiSolver<MatrixType,VectorType>::
ArnoldiSolver (const MatrixType &A, VectorType &x, int nmax, double tol_input)
:tol(tol_input)
{
	lambda.resize(nmax);
	index.resize(nmax);
	Psi.resize(nmax);
	errors.resize(nmax);
	calc_dominant(A,x);
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
calc_dominant (const MatrixType &A, VectorType &x)
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
	normalize(x0);
	do
	{
		iteration(A,x0,x); ++N_iter;
		x0 = x;
	}
	while (error>tol and N_iter<ARNOLDI_MAX_ITERATIONS);
}

tuple<complex<double>,int> find_nth_largest (int n, const VectorXcd &v)
{
	vector<tuple<complex<double>,int>> vals;
	for (int i=0; i<v.rows(); ++i) vals.push_back(make_tuple(v(i),i));
	
	sort(vals.begin(), vals.end(), [] (tuple<complex<double>,int> a, tuple<complex<double>,int> b)
	{
		return abs(get<0>(a)) > abs(get<0>(b));
	});
	
	return vals[n];
}

template<typename MatrixType, typename VectorType>
void ArnoldiSolver<MatrixType,VectorType>::
iteration (const MatrixType &A, const VectorType &x0, VectorType &x)
{
	Kbasis.clear();
	Kbasis.resize(dimK+1);
	Kbasis[0] = x0;
	normalize(Kbasis[0]);
	// overlap matrix
	MatrixXcd h(dimK+1,dimK); h.setZero();
	ComplexEigenSolver<MatrixXcd> Eugen;
	size_t max;
	
	dimKc = lambda.size(); // current Krylov dimension
	vector<complex<double> > lambda_old(lambda.size());
	for (int n=0; n<lambda.size(); ++n)
	{
		lambda_old[n] = complex<double>(1e3,1e3);
	}
	error = 1e3;
	// Arnoldi construction of an orthogonal Krylov space basis
	for (size_t k=1; k<=dimK; ++k)
	{
		HxV(A,Kbasis[k-1], Kbasis[k]);
		for (size_t j=0; j<k; ++j)
		{
			h(j,k-1) = dot(Kbasis[j],Kbasis[k]);
//			Kbasis[k] -= h(j,k-1) * Kbasis[j];
			addScale(-h(j,k-1),Kbasis[j], Kbasis[k]);
		}
		h(k,k-1) = norm(Kbasis[k]);
		Kbasis[k] /= h(k,k-1);
		
		dimKc = k;
		
		// calculate dominant eigenvectors within the Krylov space
		Eugen.compute(h.topLeftCorner(dimKc,dimKc));
		Eugen.eigenvalues().cwiseAbs().maxCoeff(&max);
		
		lambda[0] = Eugen.eigenvalues()(max);
		errors[0] = abs(lambda[0]-lambda_old[0]);
		
		for (int n=1; n<min(static_cast<int>(lambda.size()),static_cast<int>(Eugen.eigenvalues().rows())); ++n)
		{
			lambda[n] = get<0>(find_nth_largest(n,Eugen.eigenvalues()));
			index[n] = get<1>(find_nth_largest(n,Eugen.eigenvalues()));
			errors[n] = abs(lambda[n]-lambda_old[n]);
//			cout << "n=" << n << ", lambda=" << lambda[n] << ", error=" << errors[n] << endl;
		}
		error = *max_element(errors.begin(), errors.end());
		
		lambda_old = lambda;
		
		if (error < tol) {break;}
	}
	
	// project out from Krylov space
	x = Eugen.eigenvectors().col(max)(0) * Kbasis[0];
	assert(dimKc == Eugen.eigenvectors().rows() and "Bad Krylov matrix in ArndoldiSolver! Perhaps nan or all zero.");
	for (size_t k=1; k<dimKc; ++k)
	{
		//x += Eugen.eigenvectors().col(max)(k) * Kbasis[k];
		addScale(Eugen.eigenvectors().col(max)(k),Kbasis[k], x);
	}
	
	for (int n=1; n<min(static_cast<int>(lambda.size()),static_cast<int>(Eugen.eigenvalues().rows())); ++n)
	{
		Psi[n-1] = Eugen.eigenvectors().col(index[n])(0) * Kbasis[0];
		for (size_t k=1; k<dimKc; ++k)
		{
			addScale(Eugen.eigenvectors().col(index[n])(k),Kbasis[k], Psi[n-1]);
		}
	}
}

#endif

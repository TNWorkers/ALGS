#ifndef GENERICGMRESSOLVER
#define GENERICGMRESSOLVER

#ifndef GMRES_MAX_ITERATIONS
#define GMRES_MAX_ITERATIONS 1e2
#endif

template<typename MatrixType, typename VectorType>
class GMResSolver
{
public:
	
	GMResSolver(){};
	
	GMResSolver (const MatrixType &A, const VectorType &b, VectorType &x, double tol=1e-14);
	
	void solve_linear (const MatrixType &A, const VectorType &b, VectorType &x, double tol=1e-14);
	
	void set_dimK (size_t dimK_input);
	
	string info() const;
	
private:
	
	size_t dimK, dimA;
	double residual = std::nan("1");
	size_t N_iter;
	
	bool USER_HAS_FORCED_DIMK;
	
	vector<VectorType> Kbasis;
	
	void iteration (const MatrixType &A, const VectorType &b, const VectorType &x0, VectorType &x);
};

template<typename MatrixType, typename VectorType>
string GMResSolver<MatrixType,VectorType>::
info() const
{
	stringstream ss;
	
	ss << "GMResSolver" << ":"
	<< " dimA=" << dimA
	<< ", dimK=" << dimK
	<< ", iterations=" << N_iter;
	if (N_iter == GMRES_MAX_ITERATIONS)
	{
		ss << ", breakoff after max.iterations";
	}
	
	return ss.str();
}

template<typename MatrixType, typename VectorType>
GMResSolver<MatrixType,VectorType>::
GMResSolver (const MatrixType &A, const VectorType &b, VectorType &x, double tol)
{
	solve(A,b,x,tol);
}

template<typename MatrixType, typename VectorType>
void GMResSolver<MatrixType,VectorType>::
solve_linear (const MatrixType &A, const VectorType &b, VectorType &x, double tol)
{
	dimA = dim(A);
	N_iter = 0;
	
	if (!USER_HAS_FORCED_DIMK)
	{
		if      (dimA==1)             {dimK=1;}
		else if (dimA>1 and dimA<200) {dimK=static_cast<size_t>(ceil(max(2.,0.4*dimA)));}
		else                          {dimK=100;}
	}
	
	VectorType x0 = b;
//	setZero(x0);
	GaussianRandomVector<VectorType,double>::fill(dimA,x0);
	
	do
	{
		iteration(A,b,x0,x); ++N_iter;
		x0 = x;
	}
	while (residual>tol and N_iter<GMRES_MAX_ITERATIONS);
}

template<typename MatrixType, typename VectorType>
void GMResSolver<MatrixType,VectorType>::
set_dimK (size_t dimK_input)
{
	dimK = dimK_input;
	USER_HAS_FORCED_DIMK = true;
}

template<typename MatrixType, typename VectorType>
void GMResSolver<MatrixType,VectorType>::
iteration (const MatrixType &A, const VectorType &b, const VectorType &x0, VectorType &x)
{
	VectorType r0;
	HxV(A,x0, r0);
	r0 *= -1.;
	r0 += b;
	double beta = norm(r0);
	
	Kbasis.clear();
	Kbasis.resize(dimK+1);
	Kbasis[0] = r0 / beta;
	
	// overlap matrix
	MatrixXd h(dimK+1,dimK); h.setZero();
	
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
	
	// solve linear system in Krylov space
	VectorXd y = h.jacobiSvd(ComputeThinU|ComputeThinV).solve(beta*VectorXd::Unit(dimK+1,0));
	
	// a posteriori residual calculation
	residual = h(dimK,dimK-1)*abs(y(dimK-1));
	
	// project out of Krylov space
	x = x0;
	for (size_t k=0; k<dimK; ++k)
	{
		x += y(k) * Kbasis[k];
	}
}

#endif

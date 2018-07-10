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
	
	size_t get_Niter() const {return N_iterations;};
	size_t get_dimK()  const {return dimK;};
	
private:
	
	size_t dimK, dimA, dimKc;
	double residual = std::nan("1");
	double tol;
	size_t N_iterations;
	
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
	<< ", dimKmax=" << dimK
	<< ", dimK=" << dimKc
	<< ", iterations=" << N_iterations
	<< ", error=" << residual;
	if (N_iterations == GMRES_MAX_ITERATIONS)
	{
		ss << ", breakoff after max.iterations";
	}
	
	return ss.str();
}

template<typename MatrixType, typename VectorType>
GMResSolver<MatrixType,VectorType>::
GMResSolver (const MatrixType &A, const VectorType &b, VectorType &x, double tol_input)
{
	solve(A,b,x,tol);
}

template<typename MatrixType, typename VectorType>
void GMResSolver<MatrixType,VectorType>::
solve_linear (const MatrixType &A, const VectorType &b, VectorType &x, double tol_input)
{
	tol = tol_input;
	size_t try_dimA = dim(A);
	size_t try_dimb = dim(b);
	assert(try_dimA != 0 or try_dimb != 0);
	dimA = max(try_dimA, try_dimb);
	
	N_iterations = 0;
	
	if (!USER_HAS_FORCED_DIMK)
	{
		if      (dimA==1)             {dimK=1;}
		else if (dimA>1 and dimA<200) {dimK=static_cast<size_t>(ceil(max(2.,0.4*dimA)));}
		else                          {dimK=100;}
	}
	
	VectorType x0 = b;
	setZero(x0);
//	GaussianRandomVector<VectorType,double>::fill(dimA,x0);
	
	do
	{
		iteration(A,b,x0,x); ++N_iterations;
		x0 = x;
	}
	while (residual>tol and N_iterations<GMRES_MAX_ITERATIONS);
//	
//	VectorType c;
//	HxV(A,x,c);
//	c -= b;
//	cout << "norm(A*b-x)=" << norm(c) << endl;
}

template<typename MatrixType, typename VectorType>
void GMResSolver<MatrixType,VectorType>::
set_dimK (size_t dimK_input)
{
	dimK = dimK_input;
	USER_HAS_FORCED_DIMK = true;
}

//template<typename MatrixType, typename VectorType>
//void GMResSolver<MatrixType,VectorType>::
//iteration (const MatrixType &A, const VectorType &b, const VectorType &x0, VectorType &x)
//{
//	VectorType r0;
//	HxV(A,x0, r0);
//	r0 *= -1.;
//	r0 += b;
//	double beta = norm(r0);
//	
//	Kbasis.clear();
//	Kbasis.resize(dimK+1);
//	Kbasis[0] = r0 / beta;
//	
//	// overlap matrix
//	MatrixXd h(dimK+1,dimK); h.setZero();
//	
//	// Arnoldi construction of an orthogonal Krylov space basis
//	for (size_t j=0; j<dimK; ++j)
//	{
//		HxV(A,Kbasis[j], Kbasis[j+1]);
//		for (size_t i=0; i<=j; ++i)
//		{
//			h(i,j) = dot(Kbasis[i],Kbasis[j+1]);
//			Kbasis[j+1] -= h(i,j) * Kbasis[i];
//		}
//		h(j+1,j) = norm(Kbasis[j+1]);
//		Kbasis[j+1] /= h(j+1,j);
//	}
//	
//	// solve linear system in Krylov space
//	VectorXd y = h.jacobiSvd(ComputeThinU|ComputeThinV).solve(beta*VectorXd::Unit(dimK+1,0));
//	
//	// a posteriori residual calculation
//	residual = h(dimK,dimK-1)*abs(y(dimK-1));
//	
//	// project out of Krylov space
//	x = x0;
//	for (size_t k=0; k<dimK; ++k)
//	{
//		x += y(k) * Kbasis[k];
//	}
//}

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
	
	dimKc = 1; // current Krylov dimension
	VectorXd y;
	
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
		
		dimKc = j+1;
		
		// solve linear system in Krylov space
		y = h.topLeftCorner(dimKc+1,dimKc).jacobiSvd(ComputeThinU|ComputeThinV).solve(beta*VectorXd::Unit(dimKc+1,0));
		
		// a posteriori residual calculation
		residual = h(dimKc,dimKc-1)*abs(y(dimKc-1));
		
//		cout << h << endl;
//		cout << "j=" << j << ", dimKc=" << dimKc << ", h(dimKc,dimKc-1)=" << h(dimKc,dimKc-1) << ", abs(y(dimKc-1))=" << abs(y(dimKc-1)) << ", residual=" << residual << endl;
		
		if (residual < tol) {break;}
	}
	
	// project out of Krylov space
	x = x0;
	for (size_t k=0; k<dimKc; ++k)
	{
		x += y(k) * Kbasis[k];
	}
}

#endif

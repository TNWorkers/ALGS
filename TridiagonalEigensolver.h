#ifndef TRIDIAGONALEIGENSOLVER
#define TRIDIAGONALEIGENSOLVER

#include <Eigen/Dense>

/*extern "C" void dstev_ (const char* JOBZ, const int* N, double* D, double* E, double *Z, int* LDZ, double* WORK, int* INFO);*/
/*extern "C" void dstein_ (const int* N, const double* D, const double* E, const int* M, const double* W, const int* IBLOCK, const int* ISPLIT, double* Z, const int* LDZ, double* WORK, int* IWORK, int* IFAIL, int* INFO);*/
/*// compile with -llapack*/

class TridiagonalEigensolver
{
public:

	TridiagonalEigensolver (const Eigen::VectorXd &diag_input, const Eigen::VectorXd &subdiag_input);
	
	inline Eigen::VectorXd eigenvalues()  {return stored_eigenvalues;};
	inline Eigen::VectorXd eigvalzerosq() {return stored_eigvalzerosq;};
	
	void compute_eigvalzerosq (const Eigen::VectorXd &diag_input, const Eigen::VectorXd &subdiag_input);

private:
	
	Eigen::VectorXd stored_eigenvalues;
	Eigen::VectorXd stored_eigvalzerosq;
};

TridiagonalEigensolver::
TridiagonalEigensolver (const Eigen::VectorXd &diag_input, const Eigen::VectorXd &subdiag_input)
{
	int N = diag_input.rows();
	Eigen::VectorXd diag = diag_input; // input & output
	Eigen::VectorXd subdiag = subdiag_input; // input & output
	Eigen::MatrixXd WORK(2*N-2,2*N-2);
	int INFO = 0;
	dstev_((char*)"N", &N, diag.data(), subdiag.data(), NULL, &N, WORK.data(), &INFO);
	stored_eigenvalues = diag;
}

void TridiagonalEigensolver::
compute_eigvalzerosq (const Eigen::VectorXd &diag_input, const Eigen::VectorXd &subdiag_input)
{
	int N = diag_input.rows();
	stored_eigvalzerosq.resize(N);
	
	// parallel computation of the eigenvectors using the already obtained eigenvalues
	#pragma omp parallel for
	for (int i=0; i<N; ++i)
	{
		int Neig=1;
		Eigen::VectorXi IBLOCK(N); IBLOCK.setConstant(1);
		Eigen::VectorXi ISPLIT(N); ISPLIT.setConstant(N);
		Eigen::VectorXd EigVec(N);
		Eigen::VectorXd WORK(5*N);
		Eigen::VectorXi IWORK(N);
		Eigen::VectorXi IFAIL(1);
		int INFO = 0;
		
		dstein_ (&N, diag_input.data(), subdiag_input.data(), &Neig, stored_eigenvalues.segment(i,1).data(), 
		         IBLOCK.data(), ISPLIT.data(), EigVec.data(), &N, WORK.data(), IWORK.data(), IFAIL.data(), &INFO);
		stored_eigvalzerosq(i) = pow(EigVec(0),2);
	}
}

#endif

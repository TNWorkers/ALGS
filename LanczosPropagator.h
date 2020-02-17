#ifndef LANCZOS_PROPAGATOR
#define LANCZOS_PROPAGATOR

#include "LanczosSolver.h"
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

MatrixXcd exp (const MatrixXcd M)
{
	ComplexEigenSolver<MatrixXcd> Eugen(M);
	return Eugen.eigenvectors() * Eugen.eigenvalues().array().exp().matrix().asDiagonal() * Eugen.eigenvectors().inverse();
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
class LanczosPropagator : public LanczosSolver<Hamiltonian,VectorType,Scalar>
{
public:
	
	LanczosPropagator (double tol_input=1e-9, int dimK_input=1);
	
	string info() const;
	
//	template<typename InitVectorType> void t_step (const Hamiltonian &H, const InitVectorType &Vin, VectorType &Vout, double dt);
	void t_step (const Hamiltonian &H, const VectorType &Vin, VectorType &Vout, Scalar dt);
	void t_step (const Hamiltonian &H, VectorType &Vinout, Scalar dt);
	
	void t_step_fixed (const Hamiltonian &H, const VectorType &Vin, VectorType &Vout, Scalar dt, int dimK_input);
	
	inline double get_dist() {return dist;};
	
private:
	
//	void project_in (const Hamiltonian &H, const VectorType &Vin, VectorXcd &vout);
	void KrylovTimeMangler (Matrix<Scalar,Dynamic,1> &vinout, Scalar dt);
//	void project_out (const Hamiltonian &H, const VectorXcd &vin, VectorType &Vout);
	
	int dimK_original; // remember initial dimK, will be restored in a second run with the same instance if it was reduced in the previous run
	double calc_dist (Scalar dt);
	
	double tol;
	double dist;
	Scalar tstep;
};

template<typename Hamiltonian, typename VectorType, typename Scalar>
LanczosPropagator<Hamiltonian,VectorType,Scalar>::
LanczosPropagator (double tol_input, int dimK_input)
:LanczosSolver<Hamiltonian,VectorType,Scalar>(LANCZOS::REORTHO::FULL), tol(tol_input)
{
	this->set_dimK(dimK_input);
	dimK_original = this->dimK;
	this->Kbasis.resize(this->dimK);
	this->set_efficiency(LANCZOS::EFFICIENCY::TIME);
	
	this->infolabel = "LanczosPropagator";
	this->stat.last_N_iter=1; // for info()
}

//--------------<info>--------------
template<typename Hamiltonian, typename VectorType, typename Scalar>
string LanczosPropagator<Hamiltonian,VectorType,Scalar>::
info() const
{
	stringstream ss;
	ss << "LanczosPropagator:";
	ss << " |Vlhs-Vrhs|=" << dist;
	ss << " (tol=" << tol << ")";
	ss << ", mvms=" << this->stat.N_mvm;
	ss << ", dimK=" << this->dimK;
	ss << ", dt=" << tstep;
	ss << ", mem=" << round(this->stat.last_memory,3) << "GB";
	return ss.str();
}
//--------------</info>--------------

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosPropagator<Hamiltonian,VectorType,Scalar>::
KrylovTimeMangler (Matrix<Scalar,Dynamic,1> &vinout, Scalar dt)
{
	vinout = this->KrylovSolver.eigenvectors().transpose() * vinout; // V = O^T * V
	for (int i=0; i<this->dimK; ++i)
	{
//		vinout(i) *= exp(complex<double>(0.,-this->KrylovSolver.eigenvalues()(i)*dt)); // V = exp(-i*Lambda*dt) * O^T * V
		vinout(i) *= exp(this->KrylovSolver.eigenvalues()(i)*dt); // V = exp(-i*Lambda*dt) * O^T * V
	}
	vinout = this->KrylovSolver.eigenvectors() * vinout; // V = O*V * exp(-i*Lambda*dt) * O^T * V
}

// see: Christian Lubich, From Quantum to Classical Molecular Dynamics: Reduced Models and Numerical Analysis
// chapter III.2.2 Theorem 2.7 eq. (2.22), p. 94
template<typename Hamiltonian, typename VectorType, typename Scalar>
double LanczosPropagator<Hamiltonian,VectorType,Scalar>::
calc_dist (Scalar dt)
{
	int dimK = this->dimK;
	
//	MatrixXcd Mtmp1 = -1.i*dt*this->Htridiag();
	Matrix<Scalar,Dynamic,Dynamic> Mtmp1 = dt*this->Htridiag();
	Matrix<Scalar,Dynamic,Dynamic> HtridiagExp1 = (Mtmp1).exp();
//	MatrixXcd Mtmp2 = -1.i*dt*0.5*this->Htridiag();
	Matrix<Scalar,Dynamic,Dynamic> Mtmp2 = dt*0.5*this->Htridiag();
	Matrix<Scalar,Dynamic,Dynamic> HtridiagExp2 = (Mtmp2).exp();
//	MatrixXcd HtridiagExp1 = exp(-1.i*dt*this->Htridiag());
//	MatrixXcd HtridiagExp2 = exp(-1.i*dt*0.5*this->Htridiag());
	
	return abs(dt) * abs(this->next_b) * (1./6.*abs(HtridiagExp1(dimK-1,0)) + 2./3.*abs(HtridiagExp2(dimK-1,0)));
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosPropagator<Hamiltonian,VectorType,Scalar>::
t_step (const Hamiltonian &H, const VectorType &Vin, VectorType &Vout, Scalar dt)
{
	tstep = dt;
	this->setup_H(H,Vin);
	this->iteration(H,Vin);
	
	while (calc_dist(dt) >= tol)
	{
		this->calc_next_ab(H);
//		cout << "dimK=" << this->dimK << ", error: " << calc_dist(dt) << "\t" << tol << endl;
	}
	dist = calc_dist(dt);
	
	this->Krylov_diagonalize();
	
	Matrix<Scalar,Dynamic,1> vK(this->dimK);
	vK.setZero();
	vK(0) = norm(Vin);
//	this->project_in(Vin,vK);
	KrylovTimeMangler(vK,dt);
	this->project_out(vK,Vout);
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosPropagator<Hamiltonian,VectorType,Scalar>::
t_step_fixed (const Hamiltonian &H, const VectorType &Vin, VectorType &Vout, Scalar dt, int dimK_input)
{
	tstep = dt;
	this->setup_H(H,Vin);
	this->iteration(H,Vin);
	
	for (int i=0; i<dimK_input-3; ++i)
	{
		this->calc_next_ab(H);
	}
	dist = calc_dist(dt);
	
	this->Krylov_diagonalize();
	
	Matrix<Scalar,Dynamic,1> vK(this->dimK);
	vK.setZero();
	vK(0) = norm(Vin);
	KrylovTimeMangler(vK,dt);
	this->project_out(vK,Vout);
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosPropagator<Hamiltonian,VectorType,Scalar>::
t_step (const Hamiltonian &H, VectorType &Vinout, Scalar dt)
{
	VectorType Vtmp;
	t_step(H, Vinout,Vtmp, dt);
	Vinout = Vtmp;
}

#endif

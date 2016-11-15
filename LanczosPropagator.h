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

template<typename Hamiltonian, typename ComplexVectorType>
class LanczosPropagator : public LanczosSolver<Hamiltonian,ComplexVectorType,complex<double> >
{
public:
	
	LanczosPropagator (double tol_input=1e-9, int dimK_input=1);
	
	string info() const;
	
//	template<typename InitVectorType> void t_step (const Hamiltonian &H, const InitVectorType &Vin, ComplexVectorType &Vout, double dt);
	void t_step (const Hamiltonian &H, const ComplexVectorType &Vin, ComplexVectorType &Vout, double dt);
	void t_step (const Hamiltonian &H, ComplexVectorType &Vinout, double dt);
	
	void t_step_fixed (const Hamiltonian &H, const ComplexVectorType &Vin, ComplexVectorType &Vout, double dt, int dimK_input);
	
	inline double get_dist() {return dist;};
	
private:
	
//	void project_in (const Hamiltonian &H, const ComplexVectorType &Vin, VectorXcd &vout);
	void KrylovTimeMangler (VectorXcd &vinout, double dt);
//	void project_out (const Hamiltonian &H, const VectorXcd &vin, ComplexVectorType &Vout);
	
	int dimK_original; // remember initial dimK, will be restored in a second run with the same instance if it was reduced in the previous run
	double calc_dist (double dt);
	
	double tol;
	double dist;
	double tstep;
};

template<typename Hamiltonian, typename ComplexVectorType>
LanczosPropagator<Hamiltonian,ComplexVectorType>::
LanczosPropagator (double tol_input, int dimK_input)
:LanczosSolver<Hamiltonian,ComplexVectorType,complex<double> >(LANCZOS::REORTHO::FULL), tol(tol_input)
{
	this->set_dimK(dimK_input);
	dimK_original = this->dimK;
	this->Kbasis.resize(this->dimK);
	this->set_efficiency(LANCZOS::EFFICIENCY::TIME);

	this->infolabel = "LanczosPropagator";
	this->stat.last_N_iter=1; // for info()
}

//--------------<info>--------------
template<typename Hamiltonian, typename ComplexVectorType>
string LanczosPropagator<Hamiltonian,ComplexVectorType>::
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

//template<typename Hamiltonian, typename ComplexVectorType>
//void LanczosPropagator<Hamiltonian,ComplexVectorType>::
//project_in (const Hamiltonian &H, const ComplexVectorType &Vin, VectorXcd &vout)
//{
//	vout.resize(this->dimK);
//	for (int i=0; i<this->dimK; ++i)
//	{
//		vout(i) = dot(this->Kbasis[i],Vin);
//	}
//}

//template<typename Hamiltonian, typename ComplexVectorType>
//void LanczosPropagator<Hamiltonian,ComplexVectorType>::
//project_out (const Hamiltonian &H, const VectorXcd &vin, ComplexVectorType &Vout)
//{
//	Vout = vin(0) * (this->Kbasis[0]);
//	for (int i=1; i<this->dimK; ++i)
//	{
//		Vout += vin(i) * (this->Kbasis[i]);
//	}
//}

template<typename Hamiltonian, typename ComplexVectorType>
void LanczosPropagator<Hamiltonian,ComplexVectorType>::
KrylovTimeMangler (VectorXcd &vinout, double dt)
{
	vinout = this->KrylovSolver.eigenvectors().transpose() * vinout; // V = O^T * V
	for (int i=0; i<this->dimK; ++i)
	{
		vinout(i) *= exp(complex<double>(0.,-this->KrylovSolver.eigenvalues()(i)*dt)); // V = exp(-i*Lambda*dt) * O^T * V
	}
	vinout = this->KrylovSolver.eigenvectors() * vinout; // V = O*V * exp(-i*Lambda*dt) * O^T * V
}

// see: Christian Lubich, From Quantum to Classical Molecular Dynamics: Reduced Models and Numerical Analysis
// chapter III.2.2 Theorem 2.7 eq. (2.22), p. 94
template<typename Hamiltonian, typename ComplexVectorType>
double LanczosPropagator<Hamiltonian,ComplexVectorType>::
calc_dist (double dt)
{
	int dimK = this->dimK;
	
	MatrixXcd Mtmp1 = -1.i*dt*this->Htridiag();
	MatrixXcd HtridiagExp1 = (Mtmp1).exp();
	MatrixXcd Mtmp2 = -1.i*dt*0.5*this->Htridiag();
	MatrixXcd HtridiagExp2 = (Mtmp2).exp();
//	MatrixXcd HtridiagExp1 = exp(-1.i*dt*this->Htridiag());
//	MatrixXcd HtridiagExp2 = exp(-1.i*dt*0.5*this->Htridiag());
	
	return abs(dt) * abs(this->next_b) * (1./6.*abs(HtridiagExp1(dimK-1,0)) + 2./3.*abs(HtridiagExp2(dimK-1,0)));
}

//template<typename Hamiltonian, typename ComplexVectorType>
//template<typename InitVectorType>
//void LanczosPropagator<Hamiltonian,ComplexVectorType>::
//t_step (const Hamiltonian &H, const InitVectorType &Vin, ComplexVectorType &Vout, double dt)
//{
//	if (dt==0.) {Vout = complex<double>(1.,0.) * Vin;}
//	
//	if (this->dimK != dimK_original)
//	{
//		this->dimK = dimK_original;
//		this->Kbasis.resize(this->dimK);
//	}
//	
//	this->setup_H(H);
//	this->setup_ab(H,Vin);
//	this->Krylov_diagonalize();
//	
//	VectorXcd vK;
//	this->project_in(Vin,vK);
//	KrylovTimeMangler(vK,dt);
//	this->project_out(vK,Vout);
//}

template<typename Hamiltonian, typename ComplexVectorType>
void LanczosPropagator<Hamiltonian,ComplexVectorType>::
t_step (const Hamiltonian &H, const ComplexVectorType &Vin, ComplexVectorType &Vout, double dt)
{
//	if (this->dimK != dimK_original)
//	{
//		this->dimK = dimK_original;
//		this->Kbasis.resize(this->dimK);
//	}
	
	tstep = dt;
	this->setup_H(H);
	this->setup_ab(H,Vin);
	
//	cout << "dimK=" << this->dimK << ", error: " << calc_dist(dt) << "\t" << tol << endl;
	while (calc_dist(dt) >= tol)
	{
		this->calc_next_ab(H);
//		cout << "dimK=" << this->dimK << ", error: " << calc_dist(dt) << "\t" << tol << endl;
	}
	dist = calc_dist(dt);
	
	this->Krylov_diagonalize();
	
	VectorXcd vK(this->dimK);
	vK.setZero();
	vK(0) = norm(Vin);
//	this->project_in(Vin,vK);
	KrylovTimeMangler(vK,dt);
	this->project_out(vK,Vout);
}

template<typename Hamiltonian, typename ComplexVectorType>
void LanczosPropagator<Hamiltonian,ComplexVectorType>::
t_step_fixed (const Hamiltonian &H, const ComplexVectorType &Vin, ComplexVectorType &Vout, double dt, int dimK_input)
{
	tstep = dt;
	this->setup_H(H);
	this->setup_ab(H,Vin);
	
	for (int i=0; i<dimK_input-3; ++i)
	{
		this->calc_next_ab(H);
	}
	dist = calc_dist(dt);
	
	this->Krylov_diagonalize();
	
	VectorXcd vK(this->dimK);
	vK.setZero();
	vK(0) = norm(Vin);
	KrylovTimeMangler(vK,dt);
	this->project_out(vK,Vout);
}

template<typename Hamiltonian, typename ComplexVectorType>
void LanczosPropagator<Hamiltonian,ComplexVectorType>::
t_step (const Hamiltonian &H, ComplexVectorType &Vinout, double dt)
{
	ComplexVectorType Vtmp;
	t_step(H, Vinout,Vtmp, dt);
	Vinout = Vtmp;
}

#endif

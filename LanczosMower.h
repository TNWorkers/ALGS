#ifndef LANCZOS_MOWER
#define LANCZOS_MOWER

#include "LanczosSolver.h"

template<typename Hamiltonian, typename VectorType, typename Scalar>
class LanczosMower : public LanczosSolver<Hamiltonian,VectorType,Scalar>
{
public:
	
	LanczosMower (int dimK_input=10);
	
	void mow (const Hamiltonian &H, VectorType &Vinout, double threshold=1.0);
	
	inline double get_mowedWeight() {return mowedWeight;};
	
private:
	
	size_t Ncut;
	
	void KrylovEnergyTrimmer (Matrix<Scalar,Dynamic,1> &Vinout, const vector<size_t> &indices);
	
	int dimK_original;
	double mowedWeight;
};

template<typename Hamiltonian, typename VectorType, typename Scalar>
LanczosMower<Hamiltonian,VectorType,Scalar>::
LanczosMower (int dimK_input)
:LanczosSolver<Hamiltonian,VectorType,Scalar>(LANCZOS::REORTHO::FULL), Ncut(0)
{
	this->set_dimK(dimK_input);
	dimK_original = this->dimK;
	this->Kbasis.resize(this->dimK);
	this->set_efficiency(LANCZOS::EFFICIENCY::TIME);
	
	this->infolabel = "LanczosMower";
	this->stat.last_N_iter=1; // for info()
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosMower<Hamiltonian,VectorType,Scalar>::
mow (const Hamiltonian &H, VectorType &Vinout, double threshold)
{
	// reset dimK
	if (this->dimK != dimK_original)
	{
		this->dimK = dimK_original;
		this->Kbasis.resize(this->dimK);
	}
	
	this->setup_H(H);
	this->setup_ab(H,Vinout);
	this->Krylov_diagonalize();
//	Ncut = (this->KrylovSolver.eigenvalues().array() >= threshold).count();
	Ncut = (this->KrylovSolver.eigenvalues().array().abs() >= threshold).count();
	
	vector<size_t> indices;
	for (size_t i=0; i<this->KrylovSolver.eigenvalues().rows(); ++i)
	{
//		cout << i << ", E=" << this->KrylovSolver.eigenvalues()(i) << endl;
		if (abs(this->KrylovSolver.eigenvalues()(i)) >= threshold)
		{
			indices.push_back(i);
//			cout << i << ", E=" << this->KrylovSolver.eigenvalues()(i) << endl;
		}
	}
//	cout << endl;
//	cout << "Ncut=" << Ncut << ", indices.size()=" << indices.size() << ", dimK=" << this->dimK << endl;
	
	if (Ncut > 0)
	{
		Matrix<Scalar,Dynamic,1> vK;
		this->project_in(Vinout,vK);
		KrylovEnergyTrimmer(vK,indices);
		this->project_out(vK,Vinout);
	}
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosMower<Hamiltonian,VectorType,Scalar>::
KrylovEnergyTrimmer (Matrix<Scalar,Dynamic,1> &vK, const vector<size_t> &indices)
{
	Matrix<Scalar,Dynamic,Dynamic> P = Matrix<Scalar,Dynamic,Dynamic>::Identity(this->dimK,this->dimK);
	mowedWeight = 0.;
	
	for (size_t j=0; j<indices.size(); ++j)
	{
		size_t i = indices[j];
//		mowedWeight += vK(i) * conj(vK(i));
		mowedWeight += pow(abs(vK(i)),2);
//		P -= this->KrylovSolver.eigenvectors().col(this->dimK-1-i) * 
//		     this->KrylovSolver.eigenvectors().col(this->dimK-1-i).transpose(); // adjoint for complex?
		P -= this->KrylovSolver.eigenvectors().col(i) * 
		     this->KrylovSolver.eigenvectors().col(i).transpose(); // adjoint for complex?
	}
//	cout << P << endl << endl;
	vK = P * vK;
}

#endif

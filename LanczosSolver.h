#ifndef GENERICLANCZOSSOLVER
#define GENERICLANCZOSSOLVER

#ifndef LANCZOS_MEMORY_THRESHOLD
#define LANCZOS_MEMORY_THRESHOLD 6e6
#endif

#ifndef LANCZOS_MAX_ITERATIONS
#define LANCZOS_MAX_ITERATIONS 15
#endif

/// \cond
#include <math.h>
#include <sstream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
/// \endcond

using namespace Eigen;

#include "RandomVector.h"
#include "MemCalc.h"
#include "StringStuff.h" // for round()

#include "LanczosTypedefs.h"

#ifndef IS_REAL_FUNCTION
#define IS_REAL_FUNCTION
inline double isReal (double x) {return x;}
inline double isReal (complex<double> x) {return x.real();}
#endif

template<typename VectorType>
struct Eigenstate
{
	VectorType state;
	double energy;
};

template<typename Hamiltonian, typename VectorType, typename Scalar>
class LanczosSolver
{
public:
	
	LanczosSolver (LANCZOS::REORTHO::OPTION REORTHO_input = LANCZOS::REORTHO::NO, 
	               LANCZOS::CONVTEST::OPTION CONVTEST_input = LANCZOS::CONVTEST::COEFFWISE);
	
	//--------<ground or roof state>--------
	void ground (const Hamiltonian &H, Eigenstate<VectorType> &Vout, double eps_eigval=1e-7, double eps_coeff=1e-7, bool START_FROM_RANDOM=true);
	void roof (const Hamiltonian &H, Eigenstate<VectorType> &Vout, double eps_eigval=1e-7, double eps_coeff=1e-7, bool START_FROM_RANDOM=true);
	void edgeState (const Hamiltonian &H, Eigenstate<VectorType> &Vout, 
	                LANCZOS::EDGE::OPTION EDGE_input=LANCZOS::EDGE::GROUND,
	                double eps_eigval=1e-7, double eps_coeff=1e-7, bool START_FROM_RANDOM=true);
	
//	double Emin (const Hamiltonian &H, double eps_eigval=1e-7);
//	double Emax (const Hamiltonian &H, double eps_eigval=1e-7);
//	double edgeEigenvalue (const Hamiltonian &H, LANCZOS::EDGE::OPTION EDGE_input, double eps_eigval=1e-7);
	//--------</ground or roof state>--------
	
	//--------<info>--------
	virtual string info() const;
	double calc_memory (const Hamiltonian &H, MEMUNIT memunit=GB) const;
	int mvms() const {return stat.N_mvm;};
	//--------</info>--------
	
	//--------<force params>--------
	void set_efficiency (LANCZOS::EFFICIENCY::OPTION EFF_input);
	void set_dimK (int dimK_input);
	inline size_t get_dimK() {return dimK;}
	inline size_t get_iterations() {return stat.last_N_iter;}
	//--------</force params>--------
	
	//--------<projections>--------
	void project_in  (const VectorType &Vin, Matrix<Scalar,Dynamic,1> &vout);
	void project_out (const Matrix<Scalar,Dynamic,1> &vin, VectorType &Vout);
	//--------</projections>--------
	
protected:
	
	size_t dimK, dimH;
	int eigval_index;
	int invSubspace;
	double eigval;
	int convSubspace;
	double eps_coeff, eps_eigval;
	
	LANCZOS::EFFICIENCY::OPTION CHOSEN_EFFICIENCY;
	bool USER_HAS_FORCED_EFFICIENCY;
	LANCZOS::EDGE::OPTION CHOSEN_EDGE;
	bool USER_HAS_FORCED_DIMK;
	LANCZOS::CONVTEST::OPTION CHOSEN_CONVTEST;
	
	void setup_H (const Hamiltonian &H, const VectorType &V);
	int determine_dimK (size_t dimH_input) const;
	void set_eigval_index();
	double sq_test (const Hamiltonian &H, const VectorType &V, const double &lambda);
	
	void edgeStateIteration (const Hamiltonian &H, VectorType &u_out);
	
	double next_b;
	VectorType next_K;
	void calc_next_ab (const Hamiltonian &H);
	
	//--------------<Krylov space>--------------
	SelfAdjointEigenSolver<MatrixXd> KrylovSolver;
	MatrixXd Htridiag();
	void Krylov_diagonalize();
	vector<VectorType> Kbasis;
	VectorXd a, b;
	void setup_ab (const Hamiltonian &H, const VectorType &u);
	//--------------</Krylov space>--------------
	
	//--------------<reortho>--------------
	LANCZOS::REORTHO::OPTION CHOSEN_REORTHO;
	
	void reorthogonalize (int j);
	
	double eps, sqrteps, eta;
	double eps_invSubspace;
	vector<int> ReorthoBatch;
	MatrixXd ApprOverlap;
	
	void reortho_GramSchmidt (int j);
	void reortho_GramSchmidt (int j, VectorType &u_current, VectorType &u_prev, VectorType &u_preprev);
	
	void partialReortho_Simon (int j);
	bool firstStep;
	
	void partialReortho_Grcar (int j);
	void check_and_reortho_Grcar (int j, int k);
	double Hnorm;
	//--------------</reortho>--------------
	
	//--------------<stat>--------------
	LANCZOS::STAT stat;
	string infolabel;
	//--------------</stat>--------------
};

//--------------<construct>--------------
template<typename Hamiltonian, typename VectorType, typename Scalar>
LanczosSolver<Hamiltonian,VectorType,Scalar>::
LanczosSolver (LANCZOS::REORTHO::OPTION REORTHO_input, LANCZOS::CONVTEST::OPTION CONVTEST_input)
:CHOSEN_REORTHO(REORTHO_input), CHOSEN_CONVTEST(CONVTEST_input), USER_HAS_FORCED_EFFICIENCY(false), USER_HAS_FORCED_DIMK(false)
{
	eps = 2.2e-16;
	sqrteps = 1.4832397e-8;
	eta = 1.8064128e-12;
	eps_invSubspace = 1e-12;
	stat.reset();
	infolabel = "LanczosSolver";
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
set_efficiency (LANCZOS::EFFICIENCY::OPTION EFF_input)
{
	CHOSEN_EFFICIENCY = EFF_input;
	USER_HAS_FORCED_EFFICIENCY = true;
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
set_dimK (int dimK_input)
{
	dimK = dimK_input;
	USER_HAS_FORCED_DIMK = true;
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
inline int LanczosSolver<Hamiltonian,VectorType,Scalar>::
determine_dimK (std::size_t dimH_input) const
{
	if      (dimH_input==1)             {return 1;}
	else if (dimH_input>1 and dimH_input<200) {return static_cast<int>(std::ceil(std::max(2.,0.4*dimH_input)));}
	else                          {return 90;}
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
setup_H (const Hamiltonian &H, const VectorType &V)
{
	// vector space size should be calculable either from H or from V (else 0 is returned by dim function)
	size_t try_dimH = dim(H);
	size_t try_dimV = dim(V);
	assert(try_dimH != 0 or try_dimV != 0);
	dimH = max(try_dimH, try_dimV);
	
	invSubspace = 0;
	convSubspace = 0;
	
	if (USER_HAS_FORCED_EFFICIENCY == false)
	{
		if (dimH<LANCZOS_MEMORY_THRESHOLD) {CHOSEN_EFFICIENCY = LANCZOS::EFFICIENCY::TIME;}
		else                               {CHOSEN_EFFICIENCY = LANCZOS::EFFICIENCY::MEMORY;}
	}
	
	if (dimH<=200)
	{
		CHOSEN_REORTHO = LANCZOS::REORTHO::FULL;
	}
	
	if (USER_HAS_FORCED_DIMK == false)
	{
		dimK = determine_dimK(dimH);
	}
	
	if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::TIME)
	{
		Kbasis.resize(dimK);
	}
	
	if (CHOSEN_REORTHO == LANCZOS::REORTHO::GRCAR || CHOSEN_REORTHO == LANCZOS::REORTHO::SIMON)
	{
		ApprOverlap.resize(dimK,dimK);
		ApprOverlap.setZero();
	}
	
	// calculate Frobenius norm, approximately for large Hilbert space dimensions
	if (CHOSEN_REORTHO == LANCZOS::REORTHO::GRCAR)
	{
		int N_random = max(1,static_cast<int>(ceil(1e2/dimH)));
		if (N_random < dimH)
		{
			Hnorm = 0.;
			for (int i=0; i<N_random; ++i)
			{
				VectorType u;
				GaussianRandomVector<VectorType,Scalar>::fill(dimH,u);
				HxV(H,u); ++stat.N_mvm;
				Hnorm += squaredNorm(u);
			}
			Hnorm *= dimH/N_random;
		}
		else
		{
//			Hnorm = norm(H);
		}
	}
	
	int N_vec = (CHOSEN_CONVTEST == LANCZOS::CONVTEST::COEFFWISE) ? 2 : 1;
	if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::MEMORY)
	{
		N_vec += 3;
	}
	else if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::TIME)
	{
		N_vec += dimK;
	}
	stat.last_memory = N_vec * (::calc_memory<Scalar>(dimH));
}
//--------------</construct>--------------

//--------------<core algorithm>--------------
template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
edgeState (const Hamiltonian &H, Eigenstate<VectorType> &Vout, LANCZOS::EDGE::OPTION EDGE_input, double eps_eigval_input, double eps_coeff_input, bool START_FROM_RANDOM)
{
	eps_coeff = eps_coeff_input;
	eps_eigval = eps_eigval_input;
	
	CHOSEN_EDGE = EDGE_input;
	setup_H(H,Vout.state);
	set_eigval_index();
	
	int N_iter = 0;
	double err_eigval = 1.;
	double err_coeff  = 1.;
	eigval = std::numeric_limits<double>::infinity();
	
	if (START_FROM_RANDOM == true)
	{
		GaussianRandomVector<VectorType,Scalar>::fill(dimH,Vout.state);
	}
	VectorType Vnew;
	
	while (err_coeff >= eps_coeff or err_eigval >= eps_eigval)
	{
//		calc_next_ab(H);
		setup_ab(H,Vout.state);
		Krylov_diagonalize();
		edgeStateIteration(H,Vout.state);
		
//		if (N_iter==0)
//		if (N_iter==0 and sq_test(H,Vout.state)==0.)
//		{
////			if (CHOSEN_CONVTEST==LANCZOS::CONVTEST::SQ_TEST)
//			{
//				// If already close to groundstate, this test will give 0 and prevent an unnecessary second iteration.
//				double temp = sq_test(H,Vout.state);
//				if (temp <= 1.e-14)
//				{
//					err_coeff = temp;
//					err_eigval = 0.;
//				}
////				err_coeff = sq_test(H,Vout.state);
////				err_eigval = 0.;
//			}
//		}
//		else
		{
			err_eigval = std::abs(eigval-KrylovSolver.eigenvalues()(eigval_index));
//			if (CHOSEN_CONVTEST == LANCZOS::CONVTEST::COEFFWISE)
//			{
//				err_coeff = infNorm(Vout.state,Vnew);
//			}
//			else
			{
				err_coeff = sq_test(H, Vout.state, KrylovSolver.eigenvalues()(eigval_index));
			}
		}
		
		eigval = KrylovSolver.eigenvalues()(eigval_index);
		if (CHOSEN_CONVTEST==LANCZOS::CONVTEST::COEFFWISE) {Vnew = Vout.state;} // save state from last iteration
		
		// restart if eigval=nan, happens for small matrices sometimes (?)
		if (std::isnan(eigval))
		{
			++stat.N_restarts;
			GaussianRandomVector<VectorType,Scalar>::fill(dimH,Vout.state);
			err_eigval = 1.;
			err_coeff  = 1.;
		}
		
//		cout << setprecision(16) << N_iter << " err_eigval=" << err_eigval << " err_coeff=" << err_coeff << " eigval=" << eigval << endl;
//		cout << std::setprecision(16) << err_coeff << "\t" << sq_test(H,Vout.state) << "\t" << norm(H) << endl;
//		VectorType Vtmp = -Vnew;
//		cout << "alt.infNorm=" << infNorm(Vout.state,Vtmp) << endl;
		
		++N_iter;
		if (N_iter == LANCZOS_MAX_ITERATIONS)
		{
			stat.BREAK = true;
			break;
		}
	}
	
	stat.last_N_iter = N_iter;
	Vout.energy = eigval;
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
double LanczosSolver<Hamiltonian,VectorType,Scalar>::
sq_test (const Hamiltonian &H, const VectorType &Vin, const double &lambda)
{
//	VectorType Vtmp;
//	HxV(H,Vin,Vtmp); ++stat.N_mvm;
//	double sqrtVxHxHxV = norm(Vtmp); // sqrt(|<Psi|H^2|Psi>|)
//	double absVxHxV = std::abs(dot(Vin,Vtmp)); // |<Psi|H|Psi>|
//	return sqrtVxHxHxV-absVxHxV;
	
	VectorType Vtmp;
	HxV(H,Vin,Vtmp); ++stat.N_mvm;
	Vtmp -= lambda*Vin;
	return norm(Vtmp);
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
setup_ab (const Hamiltonian &H, const VectorType &u)
{
	a.resize(dimK); a.setZero();
	b.resize(dimK); b.setZero();
	
	if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::TIME)
	{
		// step: 0
		Kbasis[0] = u;
		normalize(Kbasis[0]);
		VectorType w;
		HxV(H,Kbasis[0],w); ++stat.N_mvm;
		a(0) = isReal(dot(w,Kbasis[0]));
		
		// step: 1
		w -= a(0) * Kbasis[0];
		if (dimK == 1)
		{
			next_b = norm(w);
			next_K = w/next_b;
		}
		else if (dimK>1)
		{
			b(1) = norm(w);
			if (std::abs(b(1)) < eps_invSubspace)
			{
				invSubspace = 1; // inv. subspace of size i
				stat.last_invSubspace = invSubspace; // for statistics
				dimK = invSubspace; // set dimK to inv. subspace size
				next_b = b(dimK); // save last b
				next_K = Kbasis[dimK]; // save last K-vector
				a.conservativeResize(dimK);
				b.conservativeResize(dimK);
				Kbasis.resize(dimK);
				set_eigval_index(); // reset eigval index
			}
			else
			{
				Kbasis[1] = w/b(1);
				reorthogonalize(0);
			}
		}
		
		// steps: 2 to dimK-2
		for (int i=1; i<dimK-1; ++i)
		{
			if (std::abs(b(i)) < eps_invSubspace)
			{
				invSubspace = i; // inv. subspace of size i
				stat.last_invSubspace = invSubspace; // for statistics
				dimK = invSubspace; // set dimK to inv. subspace size
				next_b = b(dimK); // save last b
				next_K = Kbasis[dimK]; // save last K-vector
				a.conservativeResize(dimK);
				b.conservativeResize(dimK);
				Kbasis.resize(dimK);
				set_eigval_index(); // reset eigval index
				break;
			}
			HxV(H,Kbasis[i],w); ++stat.N_mvm;
			a(i) = isReal(dot(w,Kbasis[i]));
			w -= a(i)*Kbasis[i] + b(i)*Kbasis[i-1];
			b(i+1) = norm(w);
			Kbasis[i+1] = w/b(i+1);
			reorthogonalize(i);
			
			// early exit:
			MatrixXd tau(i+1,i+1);
			tau.setZero();
			tau.diagonal() = a.head(i+1); // a(0) ... a(dimK-1)
			tau.diagonal<1>()  = b.segment(1,i); // b(1) ... b(dimK-1)
			tau.diagonal<-1>() = b.segment(1,i);
			SelfAdjointEigenSolver<MatrixXd> KrylovSolver(tau);
			
			double err_coeff = abs(b(i+1)) * abs(KrylovSolver.eigenvectors().col(eigval_index)(i)); // b(dimK) * |eigvec(dim_K-1)|
			double err_eigval = abs(eigval-KrylovSolver.eigenvalues()(eigval_index));
			
			eigval = KrylovSolver.eigenvalues()(eigval_index);
			
			if (err_coeff < eps_coeff and err_eigval < eps_eigval)
			{
				convSubspace = i+1;
				dimK = convSubspace;
				next_b = b(dimK);
				next_K = Kbasis[dimK];
				a.conservativeResize(dimK);
				b.conservativeResize(dimK);
				Kbasis.resize(dimK);
				set_eigval_index();
				break;
			}
		}
		// step: dimK-1
		if (invSubspace==0 and dimK>1 and convSubspace==0) // means no inv. subspace, no early convergence
		{
			HxV(H,Kbasis[dimK-1],w); ++stat.N_mvm;
			a(dimK-1) = isReal(dot(w,Kbasis[dimK-1]));
			w -= a(dimK-1)*Kbasis[dimK-1] + b(dimK-1)*Kbasis[dimK-2];
			next_b = norm(w);
			next_K = w/next_b;
		}
//		Scalar ortho_test;
//		for (int i=0; i<dimK; ++i)
//		for (int j=0; j<dimK; ++j)
//		{
//			Scalar res = dot(Kbasis[i],Kbasis[j]);
//			if (i==j) {res -= 1.;}
//			if (fabs(res) > fabs(ortho_test)) {ortho_test = res;}
//		}
//		cout << "ortho_test: " << ortho_test << endl;
	}
	else if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::MEMORY)
	{
		// step: 0
		VectorType u0 = u;
		normalize(u0);
		VectorType w;
		HxV(H,u0,w); ++stat.N_mvm;
		a(0) = isReal(dot(w,u0));
		
		// step: 1
		VectorType u1;
		if (dimK>1)
		{
			w -= a(0)*u0;
			b(1) = norm(w);
			u1 = w/b(1);
		}
		
		// steps: 2 to dimK-2
		for (int i=1; i<dimK-1; ++i)
		{
			if (fabs(b(i))<eps_invSubspace)
			{
				invSubspace=i;
				stat.last_invSubspace = invSubspace;
				break;
			}
			HxV(H,u1,w); ++stat.N_mvm;
			a(i) = isReal(dot(w,u1));
			w -= a(i)*u1 + b(i)*u0;
			b(i+1) = norm(w);
			u0 = w/b(i+1); // u0 = u_next
			swap(u0,u1);
		}
		// step: dimK-1
		if (invSubspace==0 and dimK>1)
		{
			HxV(H,u1,w); ++stat.N_mvm;
			a(dimK-1) = isReal(dot(w,u1));
		}
	}
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
calc_next_ab (const Hamiltonian &H)
{
	++dimK;
	a.conservativeResize(dimK);
	b.conservativeResize(dimK);
	b(dimK-1) = next_b;
	Kbasis.resize(dimK);
	Kbasis[dimK-1] = next_K;
	reorthogonalize(dimK-2);
	
	VectorType w;
	HxV(H,Kbasis[dimK-1],w); ++stat.N_mvm;
	a(dimK-1) = isReal(dot(w,Kbasis[dimK-1]));
	w -= a(dimK-1)*Kbasis[dimK-1] + b(dimK-1)*Kbasis[dimK-2];
	next_b = norm(w);
	next_K = w/next_b;
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
MatrixXd LanczosSolver<Hamiltonian,VectorType,Scalar>::
Htridiag()
{
	MatrixXd Mout(dimK,dimK);
	Mout.setZero();
	Mout.diagonal() = a;
	if (dimK>1)
	{
		Mout.diagonal<1>()  = b.tail(dimK-1);
		Mout.diagonal<-1>() = b.tail(dimK-1);
	}
	return Mout;
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
Krylov_diagonalize()
{
	KrylovSolver.compute(Htridiag());
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
edgeStateIteration (const Hamiltonian &H, VectorType &u_out)
{
	if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::TIME)
	{
		VectorXd c = KrylovSolver.eigenvectors().col(eigval_index);
		u_out = c(0)*Kbasis[0];
		for (int k=1; k<dimK; ++k)
		{
			u_out += c(k)*Kbasis[k];
		}
	}
	else if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::MEMORY)
	{
		VectorXd c = KrylovSolver.eigenvectors().col(eigval_index);
		u_out *= c(0);
//		u_out = c(0)*u_out;
		
		// step: 0
		VectorType u0 = u_out;
		normalize(u0);
		VectorType w;
		HxV(H,u0,w); ++stat.N_mvm;
		
		// step: 1
		VectorType u1;
		if (dimK>1)
		{
			w -= a(0)*u0;
			u1 = w/b(1);
			if (fabs(b(1))>eps_invSubspace) 
			{
				u_out += c(1)*u1;
			}
		}
		
		// steps: 2 to dimK-2
		for (int i=1; i<dimK-1; ++i)
		{
			if (fabs(b(i))<eps_invSubspace) {break;}
			HxV(H,u1,w); ++stat.N_mvm;
			w -= a(i)*u1+b(i)*u0;
			u0 = w/b(i+1); // u0 = u_next
			swap(u0,u1);
			u_out += c(i+1)*u1;
		}
	}
}

//template<typename Hamiltonian, typename VectorType, typename Scalar>
//inline double LanczosSolver<Hamiltonian,VectorType,Scalar>::
//Emin (const Hamiltonian &H, double eps_eigval)
//{
//	return edgeEigenvalue(H, LANCZOS::EDGE::GROUND, eps_eigval);
//}

//template<typename Hamiltonian, typename VectorType, typename Scalar>
//inline double LanczosSolver<Hamiltonian,VectorType,Scalar>::
//Emax (const Hamiltonian &H, double eps_eigval)
//{
//	return edgeEigenvalue(H, LANCZOS::EDGE::ROOF, eps_eigval);
//}

template<typename Hamiltonian, typename VectorType, typename Scalar>
inline void LanczosSolver<Hamiltonian,VectorType,Scalar>::
ground (const Hamiltonian &H, Eigenstate<VectorType> &Vout, double eps_eigval, double eps_coeff, bool START_FROM_RANDOM)
{
	edgeState(H, Vout, LANCZOS::EDGE::GROUND, eps_eigval, eps_coeff, START_FROM_RANDOM);
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
inline void LanczosSolver<Hamiltonian,VectorType,Scalar>::
roof (const Hamiltonian &H, Eigenstate<VectorType> &Vout, double eps_eigval, double eps_coeff, bool START_FROM_RANDOM)
{
	edgeState(H, Vout, LANCZOS::EDGE::ROOF, eps_eigval, eps_coeff, START_FROM_RANDOM);
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
set_eigval_index()
{
	if (CHOSEN_EDGE == LANCZOS::EDGE::GROUND)
	{
		eigval_index = 0;
	}
	else if (CHOSEN_EDGE == LANCZOS::EDGE::ROOF)
	{
		eigval_index = dimK-1;
	}
}
//--------------</core algorithm>--------------

//--------------<info>--------------
template<typename Hamiltonian, typename VectorType, typename Scalar>
string LanczosSolver<Hamiltonian,VectorType,Scalar>::
info() const
{
	stringstream ss;
	
	ss << infolabel << ":"
	<< " dimH=" << dimH
	<< ", dimK=" << dimK;
	
	if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::TIME)
	{
		ss << ", time-efficient";
	}
	else if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::MEMORY)
	{
		ss << ", memory-efficient";
	}
	
	ss << ", iterations=" << stat.last_N_iter
	<< ", mvms=" << stat.N_mvm
	<< ", mem=" << round(stat.last_memory,3) << "GB";
	
	if (CHOSEN_REORTHO != LANCZOS::REORTHO::NO)
	{ 
		ss << ", reorthog.=" << stat.N_reortho;
		if (CHOSEN_REORTHO == LANCZOS::REORTHO::FULL)
		{
			ss << " (full Gram-Schmidt)";
		}
		else if (CHOSEN_REORTHO == LANCZOS::REORTHO::GRCAR)
		{
			ss << " (Grcar algorithm)";
		}
		else if (CHOSEN_REORTHO == LANCZOS::REORTHO::SIMON)
		{
			ss << " (Simon algorithm)";
		}
	}
	else
	{
		ss << ", no reorthog.";
	}
	
	if (stat.last_invSubspace > 0)
	{
		ss << ", inv.subspace: b(" << stat.last_invSubspace << ")<" << eps_invSubspace;
	}
	if (stat.N_restarts > 0)
	{
		ss << ", had to restart from NaN " << stat.N_restarts << " time(s)";
	}
	if (stat.BREAK == true)
	{
		ss << ", breakoff after max.iterations";
	}
	
	return ss.str();
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
double LanczosSolver<Hamiltonian,VectorType,Scalar>::
calc_memory (const Hamiltonian &H, MEMUNIT memunit) const
{
	int N_vec = (CHOSEN_CONVTEST==LANCZOS::CONVTEST::COEFFWISE) ? 2 : 1;
	if (USER_HAS_FORCED_EFFICIENCY == false)
	{
		if (dim(H) > LANCZOS_MEMORY_THRESHOLD)
		{
			N_vec += 3;
		}
		else
		{
			if (USER_HAS_FORCED_DIMK==true)
			{
				N_vec += determine_dimK(dimK);
			}
			else
			{
				N_vec += determine_dimK(dim(H));
			}
		}
	}
	else
	{
		if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::TIME)
		{
			if (USER_HAS_FORCED_DIMK==true)
			{
				N_vec += determine_dimK(dimK);
			}
			else
			{
				N_vec += determine_dimK(dim(H));
			}
		}
		else if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::MEMORY)
		{
			N_vec += 3;
		}
	}
	return N_vec * (::calc_memory<Scalar>(dim(H),memunit));
}
//--------------</info>--------------

//--------------<reorthogonalization>--------------
template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
reorthogonalize (int j) // j = LanczosStep
{
	if      (CHOSEN_REORTHO == LANCZOS::REORTHO::FULL)  {reortho_GramSchmidt(j);}
	else if (CHOSEN_REORTHO == LANCZOS::REORTHO::GRCAR) {partialReortho_Grcar(j);}
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
reortho_GramSchmidt (int j) // j = LanczosStep
{
	assert(CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::TIME);
	for (int l=0; l<=j; ++l)
	{
		Kbasis[j+1] -= (dot(Kbasis[l],Kbasis[j+1]) / squaredNorm(Kbasis[l])) * Kbasis[l];
		++stat.N_reortho;
	}
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
inline void LanczosSolver<Hamiltonian,VectorType,Scalar>::
check_and_reortho_Grcar (int j, int k) // j = LanczosStep
{	
	if (fabs(ApprOverlap(j+1,k)) >= sqrteps)
	{
		reortho_GramSchmidt(j-1); // reortho j against all previous ones
		reortho_GramSchmidt(j);   // reortho j+1 against all previous ones
	}
	ApprOverlap(j+1,k) = 3.*eps; // reset appr. orthogonality to order of eps
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
partialReortho_Grcar (int j) // j = LanczosStep
{
	ApprOverlap.setIdentity(dimK,dimK);
	
	for (int k=1; k<j; ++k) // 1st off-diagonal
	{
		ApprOverlap(k,k-1) = 3.*eps;
	}
	
	if (j==0) // j=0, k=0
	{
		double omega_j0k0 = b(0)*ApprOverlap(0,1);
//		ApprOverlap(1,0) = (omega_j0k0 + GSL_SIGN(omega_j0k0)*2.*eps*Hnorm)/b(0);
		ApprOverlap(1,0) = (omega_j0k0 + copysign(2.*eps*Hnorm,omega_j0k0))/b(0);
		check_and_reortho_Grcar(1,0);
	}
	else // j>0, k=0
	{
		double omega_jk0 = b(0)*ApprOverlap(j,1) 
		                 + (a(0)-a(j))*ApprOverlap(j,0)
		                 - b(j-1)*ApprOverlap(j-1,0);
//		ApprOverlap(j+1,0) = (omega_jk0 + GSL_SIGN(omega_jk0)*2.*eps*Hnorm)/b(j);
			ApprOverlap(j+1,0) = (omega_jk0 + copysign(2.*eps*Hnorm,omega_jk0))/b(j);
		check_and_reortho_Grcar(j,0); // reorthogonalizes j, j+1
	}
	
	for (int k=1; k<j-1; ++k)  // k>0, j>0
	{
		double omega = b(k)*ApprOverlap(j,k+1) 
		             + (a(k)-a(j))*ApprOverlap(j,k)
		             + b(k-1)*ApprOverlap(j,k-1)
		             - b(j-1)*ApprOverlap(j-1,k);
//		ApprOverlap(j+1,k) = (omega + GSL_SIGN(omega)*2.*eps*Hnorm)/b(j);
		ApprOverlap(j+1,k) = (omega + copysign(2.*eps*Hnorm,omega))/b(j);
		check_and_reortho_Grcar(j,k); // reorthogonalizes j, j+1
	}
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
partialReortho_Simon (int j) // j = LanczosStep
{
//	boost::random::mt11213b MtEngine;
//	boost::normal_distribution<double> Psi(0,0.6);
//	boost::normal_distribution<double> Theta(0,0.3);

//	//-----------------<update approximate overlaps>-----------------	
//	ApprOverlap.setIdentity(dimK,dimK);

//	for (int k=1; k<j; ++k) // 1st off-diagonal
//	{
//		ApprOverlap(k,k-1) = eps*dimH*b(2)/b(j+1)*Psi(MtEngine);
//	}
//	
//	if (j==0) // k=0, j=0
//	{
//		ApprOverlap(1,0) = ApprOverlap(0,1) + 2.*b(1)*eps*Theta(MtEngine);
//	}
//	else  // k=0, j>0
//	{
//		ApprOverlap(j+1,0) = (b(1)*ApprOverlap(j,1)
//								+(a(0)-a(j))*ApprOverlap(j,0)
//								-b(j)*ApprOverlap(j-1,0)
//							 )/b(j+1)
//								+ eps*(b(1)+b(j+1))*Theta(MtEngine);
//	}

//	for (int k=1; k<j-1; ++k)  // k>0, j>0
//	{
//		ApprOverlap(j+1,k) = (b(k+1)*ApprOverlap(j,k+1)
//								+(a(k)-a(j))*ApprOverlap(j,k)
//								+b(k)*ApprOverlap(j,k-1)
//								-b(j)*ApprOverlap(j-1,k)
//							 )/b(j+1) 
//								+ eps*(b(k+1)+b(j+1))*Theta(MtEngine);
//	}
//	//-----------------</update approximate overlaps>-----------------
//	
//	//-----------------<get vectors to be orthogonalized>-----------------
//	if (firstStep==true)
//	{
//		for (int k=0; k<j; ++k)
//		{
//			if (fabs(ApprOverlap(j+1,k)) >= sqrteps)
//			{
//				cout << ApprOverlap(j+1,k) << endl;
//				for (int l=0; l<j; ++l)
//				{
//					if (fabs(ApprOverlap(j+1,l)) >= eta)
//					{
//						cout << "adding to ReorthoBatch: " << j+1 << "\t" << l << endl;
//						ReorthoBatch.push_back(l);
//					}
//				}
//			}
//		}
//	}
//	//-----------------</get vectors to be orthogonalized>-----------------
//	
//	//-----------------<orthogonalize>-----------------
//	if (ReorthoBatch.size()>0)
//	{
//		cout << ReorthoBatch.size() << " orthogonalizing " << j+1 << " against ";
//		for (int l=0; l<ReorthoBatch.size(); ++l)
//		{
//			cout << l << " ";
//			Kbasis[j+1] -= dot(Kbasis[l],Kbasis[j+1])/dot(Kbasis[l],Kbasis[l]) * Kbasis[l];
//		}
//		cout << endl;
//	}
//	//-----------------<orthogonalize>-----------------
//	
//	//-----------------<reset>-----------------
//	if (firstStep==true)
//	{
//		firstStep = false;
//		if (ReorthoBatch.size()>0)
//		{
//			cout << ReorthoBatch.size() << " popping" << endl;
//			ReorthoBatch.pop_back();
//		}
//		if (ReorthoBatch.size()>0)
//		{
//			cout << ReorthoBatch.size() << " erasing 0" << endl;
//			ReorthoBatch.erase(ReorthoBatch.begin());
//		}
//	}
//	else
//	{
//		firstStep = true;
//		ReorthoBatch.clear();
//	}
//	//-----------------</reset>-----------------
}
//--------------</reorthogonalization>--------------

//--------------<projections into Krylov space>--------------
template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
project_in (const VectorType &Vin, Matrix<Scalar,Dynamic,1> &vout)
{
	assert(CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::TIME);
	vout.resize(dimK);
	for (int i=0; i<dimK; ++i)
	{
		vout(i) = dot(Kbasis[i],Vin);
	}
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
project_out (const Matrix<Scalar,Dynamic,1> &vin, VectorType &Vout)
{
	assert(CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::TIME);
	Vout = vin(0) * Kbasis[0];
	for (int i=1; i<dimK; ++i)
	{
		Vout += vin(i) * Kbasis[i];
	}
}
//--------------</projections into Krylov space>--------------

#endif

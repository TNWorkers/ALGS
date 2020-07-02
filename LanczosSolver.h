#ifndef GENERICLANCZOSSOLVER
#define GENERICLANCZOSSOLVER

#ifndef LANCZOS_MEMORY_THRESHOLD
#define LANCZOS_MEMORY_THRESHOLD 6e6
#endif

#ifndef LANCZOS_MAX_ITERATIONS
#define LANCZOS_MAX_ITERATIONS 2
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
	
	LanczosSolver (LANCZOS::REORTHO::OPTION REORTHO_input = LANCZOS::REORTHO::NO);
	
	//--------<ground or roof state>--------
	void ground (const Hamiltonian &H, Eigenstate<VectorType> &Vout, double tol_eigval=1e-7, double tol_state=1e-7, bool START_FROM_RANDOM=true);
	void roof (const Hamiltonian &H, Eigenstate<VectorType> &Vout, double tol_eigval=1e-7, double tol_state=1e-7, bool START_FROM_RANDOM=true);
	void edgeState (const Hamiltonian &H, Eigenstate<VectorType> &Vout, 
	                LANCZOS::EDGE::OPTION EDGE_input=LANCZOS::EDGE::GROUND,
	                double tol_eigval=1e-7, double tol_state=1e-7, bool START_FROM_RANDOM=true);
	//--------</ground or roof state>--------
	
	//--------<info>--------
	virtual string info() const;
	double calc_memory (const Hamiltonian &H, const VectorType &V, MEMUNIT memunit=GB) const;
	int mvms() const {return stat.N_mvm;};
	inline double get_deltaE() const{return deltaE;};
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
	
//protected:
	
	size_t dimK, dimH;
	int eigval_index;
	int invSubspace, convSubspace;
	double eigval;
	double tol_state, tol_eigval;
	double err_eigval, err_state;
	
	LANCZOS::EFFICIENCY::OPTION CHOSEN_EFFICIENCY;
	bool USER_HAS_FORCED_EFFICIENCY=false;
	LANCZOS::EDGE::OPTION CHOSEN_EDGE;
	bool USER_HAS_FORCED_DIMK=false;
	
	void setup_H (const Hamiltonian &H, const VectorType &V);
	int determine_dimK (size_t dimH_input) const;
	void set_eigval_index (int limit=-1);
	double sq_test (const Hamiltonian &H, const VectorType &V, const double &lambda);
	
	void calc_eigenvector (const Hamiltonian &H, VectorType &u_out);
	
	double next_b;
	VectorType next_K;
	void calc_next_ab (const Hamiltonian &H);
	double deltaE;
	
	//--------------<Krylov space>--------------
	SelfAdjointEigenSolver<MatrixXd> KrylovSolver;
	MatrixXd Htridiag();
	void Krylov_diagonalize();
	vector<VectorType> Kbasis;
	VectorXd a, b;
	void iteration (const Hamiltonian &H, const VectorType &u);
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
LanczosSolver (LANCZOS::REORTHO::OPTION REORTHO_input)
:CHOSEN_REORTHO(REORTHO_input), USER_HAS_FORCED_EFFICIENCY(false), USER_HAS_FORCED_DIMK(false)
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
	if      (dimH_input==1)                   {return 1;}
	else if (dimH_input>1 and dimH_input<200) {return static_cast<int>(std::ceil(std::max(2.,0.4*dimH_input)));}
	else                                      {return 100;}
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
setup_H (const Hamiltonian &H, const VectorType &V)
{
	// vector space size should be calculable either from H or from V (otherwise 0 is returned by dim function)
	size_t try_dimH = dim(H);
	size_t try_dimV = dim(V);
	assert(try_dimH != 0 or try_dimV != 0);
	dimH = max(try_dimH, try_dimV);
	
	invSubspace = -1;
	convSubspace = -1;
	
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
	
	stat.last_memory = calc_memory(H,V);
}
//--------------</construct>--------------

//--------------<core algorithm>--------------
template<typename Hamiltonian, typename VectorType, typename Scalar>
inline void LanczosSolver<Hamiltonian,VectorType,Scalar>::
ground (const Hamiltonian &H, Eigenstate<VectorType> &Vout, double tol_eigval, double tol_state, bool START_FROM_RANDOM)
{
	edgeState(H, Vout, LANCZOS::EDGE::GROUND, tol_eigval, tol_state, START_FROM_RANDOM);
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
inline void LanczosSolver<Hamiltonian,VectorType,Scalar>::
roof (const Hamiltonian &H, Eigenstate<VectorType> &Vout, double tol_eigval, double tol_state, bool START_FROM_RANDOM)
{
	edgeState(H, Vout, LANCZOS::EDGE::ROOF, tol_eigval, tol_state, START_FROM_RANDOM);
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
edgeState (const Hamiltonian &H, Eigenstate<VectorType> &Vout, LANCZOS::EDGE::OPTION EDGE_input, double tol_eigval_input, double tol_state_input, bool START_FROM_RANDOM)
{
	tol_state = tol_state_input;
	tol_eigval = tol_eigval_input;
	
	CHOSEN_EDGE = EDGE_input;
	setup_H(H,Vout.state);
	set_eigval_index();
	
	int N_iter = 0;
	err_eigval = 1.;
	err_state  = 1.;
	eigval = std::numeric_limits<double>::infinity();
	
	if (START_FROM_RANDOM == true)
	{
		GaussianRandomVector<VectorType,Scalar>::fill(dimH,Vout.state);
	}
	
	while (err_state >= tol_state or err_eigval >= tol_eigval)
	{
		iteration(H,Vout.state);
		Krylov_diagonalize();
		calc_eigenvector(H,Vout.state);
		
//		err_eigval = std::abs(eigval-KrylovSolver.eigenvalues()(eigval_index));
//		err_state  = sq_test(H, Vout.state, KrylovSolver.eigenvalues()(eigval_index));
		
		eigval = KrylovSolver.eigenvalues()(eigval_index);
		
		// restart if eigval=nan
		if (std::isnan(eigval))
		{
			++stat.N_restarts;
			GaussianRandomVector<VectorType,Scalar>::fill(dimH,Vout.state);
			err_eigval = 1.;
			err_state  = 1.;
		}
		
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
	VectorType Vtmp;
	HxV(H,Vin,Vtmp); ++stat.N_mvm;
	Vtmp -= lambda*Vin;
	return norm(Vtmp);
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
set_eigval_index (int limit)
{
	if (CHOSEN_EDGE == LANCZOS::EDGE::GROUND)
	{
		eigval_index = 0;
	}
	else if (CHOSEN_EDGE == LANCZOS::EDGE::ROOF)
	{
		eigval_index = (limit==-1)? dimK-1:limit;
	}
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
iteration (const Hamiltonian &H, const VectorType &u)
{
	a.resize(dimK); a.setZero();
	b.resize(dimK); b.setZero();
	
	auto invariantSubSpaceExit = [this] (int i)
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
	};
	
	auto convergedExit = [this] (int i, const VectorType& next_K_input)
	{
		convSubspace = i+1;
		dimK = convSubspace;
		next_b = b(dimK);
//		next_K = Kbasis[dimK]; // only for LANCZOS::EFFICIENCY::TIME
		next_K = next_K_input;
		a.conservativeResize(dimK);
		b.conservativeResize(dimK);
		Kbasis.resize(dimK);
		set_eigval_index();
	};
	
	if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::TIME)
	{
		// step: 0
		Kbasis[0] = u;
		normalize(Kbasis[0]);
		VectorType w;
		HxV(H,Kbasis[0],w); ++stat.N_mvm;
		a(0) = isReal(dot(w,Kbasis[0]));
		
		deltaE = abs(dot(w,w)-pow(a(0),2));
//		cout << "dot(w,w)=" << dot(w,w) << ", pow(a(0),2)=" << pow(a(0),2) << endl;
		
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
				invariantSubSpaceExit(1);
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
				invariantSubSpaceExit(i);
				break;
			}
			HxV(H,Kbasis[i],w); ++stat.N_mvm;
			a(i) = isReal(dot(w,Kbasis[i]));
			w -= a(i) * Kbasis[i] + b(i) * Kbasis[i-1];
			b(i+1) = norm(w);
			Kbasis[i+1] = w/b(i+1);
			reorthogonalize(i);
			
			// Early exit due to convergence:
			SelfAdjointEigenSolver<MatrixXd> KrylovSolver(Htridiag().topLeftCorner(i+1,i+1));
			set_eigval_index(i);
			// error is: b(dimK) * |eigvec(dim_K-1)|
			err_state  = abs(b(i+1)) * abs(KrylovSolver.eigenvectors().col(eigval_index)(i)); 
			err_eigval = abs(eigval-KrylovSolver.eigenvalues()(eigval_index));
			eigval = KrylovSolver.eigenvalues()(eigval_index);
			
//			cout << "err_state=" << err_state << ", err_eigval=" << err_eigval << ", tol_state=" << tol_state << ", tol_eigval=" << tol_eigval << endl;
			
			if (err_state < tol_state and err_eigval < tol_eigval)
			{
				convergedExit(i,Kbasis[i+1]);
				break;
			}
		}
		// step: dimK-1
		// if no inv. subspace and no early convergence -> continue
		if (invSubspace == -1 and dimK > 1 and convSubspace == -1)
		{
			HxV(H,Kbasis[dimK-1],w); ++stat.N_mvm;
			a(dimK-1) = isReal(dot(w,Kbasis[dimK-1]));
			w -= a(dimK-1) * Kbasis[dimK-1] + b(dimK-1) * Kbasis[dimK-2];
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
		if (dimK > 1)
		{
			w -= a(0) * u0;
			b(1) = norm(w);
			u1 = w/b(1);
		}
		
		// steps: 2 to dimK-2
		for (int i=1; i<dimK-1; ++i)
		{
			if (fabs(b(i)) < eps_invSubspace)
			{
				invariantSubSpaceExit(i);
				break;
			}
			HxV(H,u1,w); ++stat.N_mvm;
			a(i) = isReal(dot(w,u1));
			w -= a(i) * u1 + b(i) * u0;
			b(i+1) = norm(w);
			u0 = w/b(i+1); // u0 = u_next
			swap(u0,u1);
			
			// Early exit due to convergence:
			SelfAdjointEigenSolver<MatrixXd> KrylovSolver(Htridiag().topLeftCorner(i+1,i+1));
			set_eigval_index(i);
			// error is: b(dimK) * |eigvec(dim_K-1)|
			err_state  = abs(b(i+1)) * abs(KrylovSolver.eigenvectors().col(eigval_index)(i)); 
			err_eigval = abs(eigval-KrylovSolver.eigenvalues()(eigval_index));
			eigval = KrylovSolver.eigenvalues()(eigval_index);
			
			if (err_state < tol_state and err_eigval < tol_eigval)
			{
				convergedExit(i,u1);
				break;
			}
		}
		// step: dimK-1
		// if no inv. subspace and no early convergence -> continue
		if (invSubspace == -1 and dimK > 1 and convSubspace == -1)
		{
			HxV(H,u1,w); ++stat.N_mvm;
			a(dimK-1) = isReal(dot(w,u1));
		}
	}
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
Krylov_diagonalize()
{
	KrylovSolver.compute(Htridiag());
//	cout << "e0=" << KrylovSolver.eigenvalues()(0) << endl;
//	if (KrylovSolver.eigenvalues().rows()>1)
//	{
//		cout << "e1=" << KrylovSolver.eigenvalues()(1) << endl;
//	}
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
calc_eigenvector (const Hamiltonian &H, VectorType &u_out)
{
	if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::TIME)
	{
		VectorXd c = KrylovSolver.eigenvectors().col(eigval_index);
		u_out = c(0) * Kbasis[0];
		for (int k=1; k<dimK; ++k)
		{
			u_out += c(k) * Kbasis[k];
		}
	}
	else if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::MEMORY)
	{
		VectorXd c = KrylovSolver.eigenvectors().col(eigval_index);
		u_out *= c(0);
		
		// step: 0
		VectorType u0 = u_out;
		normalize(u0);
		VectorType w;
		HxV(H,u0,w); ++stat.N_mvm;
		
		// step: 1
		VectorType u1;
		if (dimK > 1)
		{
			w -= a(0) * u0;
			u1 = w/b(1);
			if (fabs(b(1)) > eps_invSubspace) 
			{
				u_out += c(1) * u1;
			}
		}
		
		// steps: 2 to dimK-2
		for (int i=1; i<dimK-1; ++i)
		{
//			if (fabs(b(i)) < eps_invSubspace) {break;} // dimK should be already reset -> must check
			HxV(H,u1,w); ++stat.N_mvm;
			w -= a(i) * u1+b(i) * u0;
			u0 = w / b(i+1); // u0 = u_next
			swap(u0,u1);
			u_out += c(i+1) * u1;
		}
	}
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
calc_next_ab (const Hamiltonian &H)
{
//	double t_HxV = 0.;
//	double t_pre = 0.;
//	double t_post = 0.;
//	double t_tot = 0.;
//	
//	Stopwatch<> Watch_tot;
//	Stopwatch<> Watch1;
	
	++dimK;
	a.conservativeResize(dimK);
	b.conservativeResize(dimK);
	b(dimK-1) = next_b;
	Kbasis.resize(dimK);
	Kbasis[dimK-1] = next_K;
	reorthogonalize(dimK-2);
	
	VectorType w;
//	t_pre += Watch1.time(SECONDS);
//	Stopwatch<> Watch2;
	HxV(H,Kbasis[dimK-1],w); ++stat.N_mvm;
//	t_HxV += Watch2.time(SECONDS);
//	Stopwatch<> Watch3;
	a(dimK-1) = isReal(dot(w,Kbasis[dimK-1]));
	w -= a(dimK-1) * Kbasis[dimK-1] + b(dimK-1) * Kbasis[dimK-2];
	next_b = norm(w);
	next_K = w/next_b;
//	t_post += Watch3.time(SECONDS);
//	t_tot += Watch_tot.time(SECONDS);
//	
//	cout << "pre=" << t_pre/t_tot << ", HxV=" << t_HxV/t_tot << ", t_post=" << t_post/t_tot << endl;
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
	<< ", mem=" << round(stat.last_memory,3) << "GB"
	<< ", err_eigval=" << err_eigval
	<< ", err_state=" << err_state;
	
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
calc_memory (const Hamiltonian &H, const VectorType &V, MEMUNIT memunit) const
{
	size_t try_dimH = dim(H);
	size_t try_dimV = dim(V);
	size_t dimH = max(try_dimH, try_dimV);
	
	int N_vec = 1;
	
	if (USER_HAS_FORCED_EFFICIENCY == false)
	{
		if (dimH > LANCZOS_MEMORY_THRESHOLD)
		{
			N_vec += 3;
		}
		else
		{
			N_vec += dimK;
		}
	}
	else
	{
		if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::TIME)
		{
			N_vec += dimK;
		}
		else if (CHOSEN_EFFICIENCY == LANCZOS::EFFICIENCY::MEMORY)
		{
			N_vec += 3;
		}
	}
	return N_vec * (::calc_memory<Scalar>(dimH,memunit));
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
	ApprOverlap(j+1,k) = 3. * eps; // reset appr. orthogonality to order of eps
}

template<typename Hamiltonian, typename VectorType, typename Scalar>
void LanczosSolver<Hamiltonian,VectorType,Scalar>::
partialReortho_Grcar (int j) // j = LanczosStep
{
	ApprOverlap.setIdentity(dimK,dimK);
	
	for (int k=1; k<j; ++k) // 1st off-diagonal
	{
		ApprOverlap(k,k-1) = 3. * eps;
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

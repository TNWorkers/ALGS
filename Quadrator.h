#ifndef QUADRATOR
#define QUADRATOR

#include <map>
#include <Eigen/Dense>
#include <vector>
// needed only for Clenshaw-Curtis:
//#include <fftw3.h> // compile with -lfftw3 -lfftw3_omp

#include "ChebyshevAbscissa.h"
#include "TridiagonalEigensolver.h" // compile with -llapack
using namespace std;

int order (int n)
{
	if      (n<0)  {return 0;}
	else if (n==0) {return 1;}
	else           {return pow(2,n)+1;}
}

enum QUAD_RULE {GAUSS_CHEBYSHEV, GAUSS_CHEBYSHEV_LOBATTO, CLENSHAW_CURTIS, 
                GAUSS_LEGENDRE, GAUSS_HERMITE, GAUSS_LAGUERRE, 
                NEWTON_COTES_CLOSED, NEWTON_COTES_OPEN, FILON_CLENSHAW_CURTIS};

template<QUAD_RULE qrule>
class Quadrator
{
public:
	
	Quadrator(){};
	Quadrator(double b_input)
	:b(b_input)
	{};
	
	double abscissa (int k, double xmin, double xmax, int xpoints);
	double abscissa (int k, int xpoints);
	double phase (int k, double xmin, double xmax, int xpoints);
	
	double weight (int k, double xmin, double xmax, int xpoints);
	double weight (int k, int xpoints);
	double xp_weight (int k, double xmin, double xmax, int xpoints);
	
	int order (int n);
	
	Eigen::ArrayXd get_abscissa (double xmin, double xmax, int xpoints)
	{
		Eigen::VectorXd Vout(xpoints);
		for (int i=0; i<xpoints; ++i)
		{
			Vout(i) = abscissa(i,xmin,xmax,xpoints);
		}
		return Vout;
	}
	
	Eigen::ArrayXd get_steps (double xmin, double xmax, int xpoints)
	{
		Eigen::VectorXd x = get_abscissa(xmin,xmax,xpoints);
		
		Eigen::VectorXd Vout(xpoints);
		Vout(0) = x(0);
		for (int i=1; i<xpoints; ++i)
		{
			Vout(i) = x(i)-x(i-1);
		}
		
		return Vout;
	}
	
	Eigen::ArrayXd get_weights (double xmin, double xmax, int xpoints)
	{
		Eigen::VectorXd Vout(xpoints);
		for (int i=0; i<xpoints; ++i)
		{
			Vout(i) = weight(i,xmin,xmax,xpoints);
		}
		return Vout;
	}
	
	double integrate (double (*f)(double), double xmin, double xmax, int xpoints);
	double integrate (double (*f)(double,double), double xmin, double xmax, double ymin, double ymax, int xpoints);
	double integrate (double (*f)(double,double,double), double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, int xpoints);
	double infigrate (double (*f)(double), int xpoints);
	
	// for compatibility with the GSL
//	double integrate (double (*f)(double,void*), double xmin, double xmax, int xpoints);
//	double infigrate (double (*f)(double,void*), int xpoints);
	
	double diffigrate (double (*f)(double,double,double), double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, const Eigen::VectorXi indexSet);
	double diff_xp_weight (int k, double xmin, double xmax, int order);
	double diff_weight (int k, double xmin, double xmax, int order);
	int diff_feval (const Eigen::VectorXi indexSet);
	
private:
	
	double b=1.;
	
	vector<Eigen::VectorXd> weights;
	vector<Eigen::VectorXd> points;
	map<int,int> IndexDictionary; // dictionary of indices: xpoints -> position in vectorsweights & points
	
	void calculate_weights (int xpoints);
};

template<QUAD_RULE qrule>
inline int Quadrator<qrule>::
order (int n)
{
	if      (n<0)  {return 0;}
	else if (n==0) {return 1;}
	else           {return pow(2,n)+1;}
}

template<QUAD_RULE qrule>
inline double Quadrator<qrule>::
weight (int k, double xmin, double xmax, int xpoints)
{
	return 0;
}

template<QUAD_RULE qrule>
inline double Quadrator<qrule>::
xp_weight (int k, double xmin, double xmax, int xpoints)
{
	return 0;
}

template<QUAD_RULE qrule>
inline double Quadrator<qrule>::
diff_xp_weight (int k, double xmin, double xmax, int xpoints)
{
	return 0;
}

template<QUAD_RULE qrule>
inline double Quadrator<qrule>::
phase (int k, double xmin, double xmax, int xpoints)
{
	return 0;
}

template<QUAD_RULE qrule>
double Quadrator<qrule>::
integrate (double (*f)(double), double xmin, double xmax, int xpoints)
{
	double res = 0.;
	#ifndef QUADRATOR_DONT_USE_OPENMP
	#pragma omp parallel for reduction(+:res)
	#endif
	for (int ix=0; ix<xpoints; ++ix)
	{
		res += weight(ix,xmin,xmax,xpoints) * f(abscissa(ix,xmin,xmax,xpoints));
	}
	return res;
}

template<QUAD_RULE qrule>
double Quadrator<qrule>::
infigrate (double (*f)(double), int xpoints)
{
	double res = 0.;
	#ifndef QUADRATOR_DONT_USE_OPENMP
	#pragma omp parallel for reduction(+:res)
	#endif
	for (int ix=0; ix<xpoints; ++ix)
	{
//		// rescaling 1:
//		double t = abscissa(ix,xpoints);
//		res += weight(ix,xpoints) * f(t/(1.-t*t)) * (1.+t*t)/pow(1.-t*t,2.);

		// rescaling 2 (much better!):
		double t = abscissa(ix,0.,1.,xpoints);
		res += weight(ix,0.,1.,xpoints) * ( f((1.-t)/t) + f((t-1.)/t) )/pow(t,2);
	}
	return res;
}

template<QUAD_RULE qrule>
double Quadrator<qrule>::
integrate (double (*f)(double,double), double xmin, double xmax, double ymin, double ymax, int xpoints)
{
	double res = 0.;
	#ifndef QUADRATOR_DONT_USE_OPENMP
	#pragma omp parallel for collapse(2) reduction(+:res)
	#endif
	for (int ix=0; ix<xpoints; ++ix)
	for (int iy=0; iy<xpoints; ++iy)
	{
		res += weight(ix,xmin,xmax,xpoints)
		      *weight(iy,ymin,ymax,xpoints) 
			  *f(abscissa(ix,xmin,xmax,xpoints),
			     abscissa(iy,ymin,ymax,xpoints));
	}
	return res;
}

template<QUAD_RULE qrule>
double Quadrator<qrule>::
integrate (double (*f)(double,double,double), double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, int xpoints)
{
	double res = 0.;
	#ifndef QUADRATOR_DONT_USE_OPENMP
	#pragma omp parallel for collapse(3) reduction(+:res)
	#endif
	for (int ix=0; ix<xpoints; ++ix)
	for (int iy=0; iy<xpoints; ++iy)
	for (int iz=0; iz<xpoints; ++iz)
	{
		res += weight(ix,xmin,xmax,xpoints)
		      *weight(iy,ymin,ymax,xpoints)
		      *weight(iz,zmin,zmax,xpoints)
			  *f(abscissa(ix,xmin,xmax,xpoints),
			     abscissa(iy,ymin,ymax,xpoints),
			     abscissa(iz,zmin,zmax,xpoints));
	}
	return res;
}

// not very efficient; for non-nested rules
template<QUAD_RULE qrule>
double Quadrator<qrule>::
diffigrate (double (*f)(double,double,double), double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, const Eigen::VectorXi indexSet)
{
	assert(indexSet.rows()==3);
	
	int Np1 = order(indexSet(0));
	int Np2 = order(indexSet(1));
	int Np3 = order(indexSet(2));
	
	int Mp1 = order(indexSet(0)-1);
	int Mp2 = order(indexSet(1)-1);
	int Mp3 = order(indexSet(2)-1);
	
	Eigen::VectorXd px(Np1+Mp1);
	Eigen::VectorXd py(Np2+Mp2);
	Eigen::VectorXd pz(Np3+Mp3);
	
	Eigen::VectorXd wx(Np1+Mp1);
	Eigen::VectorXd wy(Np2+Mp2);
	Eigen::VectorXd wz(Np3+Mp3);
	
	for (int i=0; i<Np1; ++i)
	{
		px(i) = abscissa(i,xmin,xmax,Np1);
		wx(i) = weight(i,xmin,xmax,Np1);
	}
	for (int i=0; i<Np2; ++i)
	{
		py(i) = abscissa(i,ymin,ymax,Np2);
		wy(i) = weight(i,ymin,ymax,Np2);
	}
	for (int i=0; i<Np3; ++i)
	{
		pz(i) = abscissa(i,zmin,zmax,Np3);
		wz(i) = weight(i,zmin,zmax,Np3);
	}
	
	for (int i=0; i<Mp1; ++i)
	{
		px(Np1+i) = abscissa(i,xmin,xmax,Mp1);
		wx(Np1+i) = -weight(i,xmin,xmax,Mp1);
	}
	for (int i=0; i<Mp2; ++i)
	{
		py(Np2+i) = abscissa(i,ymin,ymax,Mp2);
		wy(Np2+i) = -weight(i,ymin,ymax,Mp2);
	}
	for (int i=0; i<Mp3; ++i)
	{
		pz(Np3+i) = abscissa(i,zmin,zmax,Mp3);
		wz(Np3+i) = -weight(i,zmin,zmax,Mp3);
	}
	
	double res = 0;
	#ifndef QUADRATOR_DONT_USE_OPENMP
	#pragma omp parallel for collapse(3) reduction(+:res)
	#endif
	for (int ix=0; ix<px.rows(); ++ix)
	for (int iy=0; iy<py.rows(); ++iy)
	for (int iz=0; iz<pz.rows(); ++iz)
	{
		res += wx(ix)*wy(iy)*wz(iz) * f(px(ix),py(iy),pz(iz));
	}
	
	return res;
}

// default: non-nested
template<QUAD_RULE qrule>
inline int Quadrator<qrule>::
diff_feval (const Eigen::VectorXi indexSet)
{
	int res = 1;
	for (int i=0; i<indexSet.rows(); ++i)
	{
		res *= order(indexSet(i)) + order(indexSet(i)-1);
	}
	return res;
}

//-----------------------------<Chebyshev-Gauss>---------------------------
template<>
inline double Quadrator<GAUSS_CHEBYSHEV>::
abscissa (int k, double xmin, double xmax, int xpoints)
{
	return ChebyshevAbscissa(k,xmin,xmax,xpoints);
}

template<>
inline double Quadrator<GAUSS_CHEBYSHEV>::
abscissa (int k, int xpoints)
{
	return ChebyshevAbscissa(k,xpoints);
}

template<>
inline double Quadrator<GAUSS_CHEBYSHEV>::
weight (int k, double xmin, double xmax, int xpoints)
{
	return ChebyshevWeight(k,xmin,xmax,xpoints);
}

template<>
inline double Quadrator<GAUSS_CHEBYSHEV>::
weight (int k, int xpoints)
{
	return ChebyshevWeight(k,xpoints);
}

template<>
inline double Quadrator<GAUSS_CHEBYSHEV>::
xp_weight (int k, double xmin, double xmax, int xpoints)
{
	return ChebyshevXpWeight(k,xmin,xmax,xpoints);
}

template<>
inline double Quadrator<GAUSS_CHEBYSHEV>::
phase (int k, double xmin, double xmax, int xpoints)
{
	return ChebyshevPhase(k,xmin,xmax,xpoints);
}
//-----------------------------</Chebyshev-Gauss>---------------------------

//-----------------------------<Chebyshev-Gauss-Lobatto>---------------------------
template<>
inline double Quadrator<GAUSS_CHEBYSHEV_LOBATTO>::
abscissa (int k, double xmin, double xmax, int xpoints)
{
	return ChebyshevLobattoAbscissa (k,xmin,xmax,xpoints);
}

template<>
inline double Quadrator<GAUSS_CHEBYSHEV_LOBATTO>::
abscissa (int k, int xpoints)
{
	return ChebyshevLobattoAbscissa(k,xpoints);
}

template<>
inline double Quadrator<GAUSS_CHEBYSHEV_LOBATTO>::
weight (int k, double xmin, double xmax, int xpoints)
{
	return ChebyshevLobattoWeight(k,xmin,xmax,xpoints);
}

template<>
inline double Quadrator<GAUSS_CHEBYSHEV_LOBATTO>::
weight (int k, int xpoints)
{
	return ChebyshevLobattoWeight(k,xpoints);
}

template<>
inline double Quadrator<GAUSS_CHEBYSHEV_LOBATTO>::
xp_weight (int k, double xmin, double xmax, int xpoints)
{
	return ChebyshevLobattoXpWeight(k,xmin,xmax,xpoints);
}

template<>
inline double Quadrator<GAUSS_CHEBYSHEV_LOBATTO>::
phase (int k, double xmin, double xmax, int xpoints)
{
	return ChebyshevLobattoPhase(k,xmin,xmax,xpoints);
}

// differential integral
template<>
inline double Quadrator<GAUSS_CHEBYSHEV_LOBATTO>::
diff_weight (int k, double xmin, double xmax, int order)
{
	return ChebyshevLobattoDiffWeight(k,xmin,xmax,order);
}

// differential integral of Chebyshev expansion
template<>
inline double Quadrator<GAUSS_CHEBYSHEV_LOBATTO>::
diff_xp_weight (int k, double xmin, double xmax, int order)
{
	return ChebyshevLobattoDiffXpWeight(k,xmin,xmax,order);
}

template<>
double Quadrator<GAUSS_CHEBYSHEV_LOBATTO>::
diffigrate (double (*f)(double,double,double), double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, const Eigen::VectorXi indexSet)
{
	assert(indexSet.rows()==3);
	
	int Np1 = order(indexSet(0));
	int Np2 = order(indexSet(1));
	int Np3 = order(indexSet(2));
	
	double res = 0.;
	
	#ifndef QUADRATOR_DONT_USE_OPENMP
	#pragma omp parallel for collapse(3) reduction(+:res)
	#endif
	for (int ix=0; ix<Np1; ++ix)
	for (int iy=0; iy<Np2; ++iy)
	for (int iz=0; iz<Np3; ++iz)
	{
		res += ChebyshevLobattoDiffWeight(ix, xmin,xmax, indexSet(0))*
		       ChebyshevLobattoDiffWeight(iy, ymin,ymax, indexSet(1))*
		       ChebyshevLobattoDiffWeight(iz, zmin,zmax, indexSet(2))
			  *f(ChebyshevLobattoAbscissa(ix, xmin,xmax, Np1),
			     ChebyshevLobattoAbscissa(iy, ymin,ymax, Np2),
			     ChebyshevLobattoAbscissa(iz, zmin,zmax, Np3)
			    );
	}
	return res;
}

template<>
inline int Quadrator<GAUSS_CHEBYSHEV_LOBATTO>::
diff_feval (const Eigen::VectorXi indexSet)
{
	int res = 1;
	for (int i=0; i<indexSet.rows(); ++i)
	{
		res *= order(indexSet(i));
	}
	return res;
}
//-----------------------------</Chebyshev-Gauss-Lobatto>---------------------------

//-----------------------------<Legendre-Gauss>---------------------------
template<>
void Quadrator<GAUSS_LEGENDRE>::
calculate_weights (int xpoints)
{
//	// using Eigen
//	Eigen::MatrixXd Jacobi(xpoints,xpoints);
//	Jacobi.setZero();
//	for (int k=0; k<xpoints-1; ++k)
//	{
//		Jacobi(k,k+1) = 1./sqrt(4.-pow(k+1,-2.));
//		Jacobi(k+1,k) = Jacobi(k,k+1);
//	}
//	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Eugen(Jacobi);
//	points.push_back(Eugen.eigenvalues());
//	points.push_back(Eugen.eigenvectors().row(0).array().square().matrix()); // note: weights have to be multiplied by mu0
	
	// using LAPACK <- faster
	Eigen::VectorXd Jacobi_diag(xpoints);
	Jacobi_diag.setZero();
	Eigen::VectorXd Jacobi_subdiag(xpoints-1);
	for (int n=0; n<xpoints-1; ++n)
	{
		Jacobi_subdiag(n) = 1./sqrt(4.-pow(n+1,-2.));
	}
	
	TridiagonalEigensolver Tim(Jacobi_diag, Jacobi_subdiag);
	Tim.compute_eigvalzerosq(Jacobi_diag, Jacobi_subdiag); // need 0-th row squared
	
	IndexDictionary.insert(std::pair<int,int>(xpoints,IndexDictionary.size()));
	points.push_back(Tim.eigenvalues());
	weights.push_back(Tim.eigvalzerosq());
}

template<>
inline double Quadrator<GAUSS_LEGENDRE>::
abscissa (int k, int xpoints)
{
//	if (points.rows()!=xpoints) {calculate_weights(xpoints);}
//	return points(k);
	#pragma omp critical
	{
		if (IndexDictionary.find(xpoints) == IndexDictionary.end()) {calculate_weights(xpoints);}
	}
	return points[IndexDictionary[xpoints]](k);
}

template<>
inline double Quadrator<GAUSS_LEGENDRE>::
abscissa (int k, double xmin, double xmax, int xpoints)
{
//	if (points.rows()!=xpoints) {calculate_weights(xpoints);}
	#pragma omp critical
	{
		if (IndexDictionary.find(xpoints) == IndexDictionary.end()) {calculate_weights(xpoints);}
	}
	double a = (xmax-xmin)*0.5;
	double b = (xmax+xmin)*0.5;
//	return a*points(k)+b;
	return a*points[IndexDictionary[xpoints]](k)+b;
}

template<>
inline double Quadrator<GAUSS_LEGENDRE>::
weight (int k, int xpoints)
{
	if (xpoints==1) {return 2.;}
//	if (weights.rows()!=xpoints) {calculate_weights(xpoints);}
	#pragma omp critical
	{
		if (IndexDictionary.find(xpoints) == IndexDictionary.end()) {calculate_weights(xpoints);}
	}
//	return 2.*weights(k);
	return 2.*weights[IndexDictionary[xpoints]](k);
}

template<>
inline double Quadrator<GAUSS_LEGENDRE>::
weight (int k, double xmin, double xmax, int xpoints)
{
	if (xpoints==1) {return xmax-xmin;}
//	if (weights.rows()!=xpoints) {calculate_weights(xpoints);}
	#pragma omp critical
	{
		if (IndexDictionary.find(xpoints) == IndexDictionary.end()) {calculate_weights(xpoints);}
	}
//	return (xmax-xmin)*weights(k);
	return (xmax-xmin)*weights[IndexDictionary[xpoints]](k);
}
//-----------------------------</Legendre-Gauss>---------------------------

//-----------------------------<Laguerre-Gauss>---------------------------
template<>
void Quadrator<GAUSS_LAGUERRE>::
calculate_weights (int xpoints)
{
//	// using Eigen
//	Eigen::MatrixXd Jacobi(xpoints,xpoints);
//	Jacobi.setZero();
//	for (int k=0; k<xpoints-1; ++k)
//	{
//		Jacobi(k,k+1) = k+1;
//		Jacobi(k+1,k) = Jacobi(k,k+1);
//	}
//	for (int k=0; k<xpoints; ++k)
//	{
//		Jacobi(k,k) = 2*(k+1)-1;
//	}
//	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Eugen(Jacobi);
//	points.push_back(Eugen.eigenvalues());
//	weights.push_back(Eugen.eigenvectors().row(0).array().square().matrix()); // note: weights have to be multiplied by mu0
	
	// using LAPACK
	Eigen::VectorXd Jacobi_diag(xpoints);
	for (int k=0; k<xpoints; ++k)
	{
		Jacobi_diag(k) = 2*(k+1)-1;
	}
	Eigen::VectorXd Jacobi_subdiag(xpoints-1);
	for (int k=0; k<xpoints-1; ++k)
	{
		Jacobi_subdiag(k) = k+1;
	}
	
	TridiagonalEigensolver Tim(Jacobi_diag, Jacobi_subdiag);
	Tim.compute_eigvalzerosq(Jacobi_diag, Jacobi_subdiag); // need 0-th row squared
	
	IndexDictionary.insert(std::pair<int,int>(xpoints,IndexDictionary.size()));
	points.push_back(Tim.eigenvalues());
	weights.push_back(Tim.eigvalzerosq());
}

template<>
inline double Quadrator<GAUSS_LAGUERRE>::
abscissa (int k, int xpoints)
{
	#pragma omp critical
	{
		if (IndexDictionary.find(xpoints) == IndexDictionary.end()) {calculate_weights(xpoints);}
	}
	return points[IndexDictionary[xpoints]](k)/b;
}

template<>
inline double Quadrator<GAUSS_LAGUERRE>::
weight (int k, int xpoints)
{
	#pragma omp critical
	{
		if (IndexDictionary.find(xpoints) == IndexDictionary.end()) {calculate_weights(xpoints);}
	}
	return weights[IndexDictionary[xpoints]](k)/b;
}
//-----------------------------</Laguerre-Gauss>---------------------------

//-----------------------------<Hermite-Gauss>---------------------------
template<>
void Quadrator<GAUSS_HERMITE>::
calculate_weights (int xpoints)
{
	Eigen::VectorXd Jacobi_diag(xpoints);
	Jacobi_diag.setZero();
	Eigen::VectorXd Jacobi_subdiag(xpoints-1);
	for (int n=0; n<xpoints-1; ++n)
	{
		Jacobi_subdiag(n) = M_SQRT1_2*sqrt(n+1);
	}
	
	TridiagonalEigensolver Tim(Jacobi_diag, Jacobi_subdiag);
	Tim.compute_eigvalzerosq(Jacobi_diag, Jacobi_subdiag);

	IndexDictionary.insert(std::pair<int,int>(xpoints,IndexDictionary.size()));
	points.push_back(Tim.eigenvalues());
	weights.push_back(M_SQRTPI*Tim.eigvalzerosq());
}

template<>
inline double Quadrator<GAUSS_HERMITE>::
abscissa (int k, int xpoints)
{
	#pragma omp critical
	{
		if (IndexDictionary.find(xpoints) == IndexDictionary.end()) {calculate_weights(xpoints);}
	}
	return points[IndexDictionary[xpoints]](k);
}

template<>
inline double Quadrator<GAUSS_HERMITE>::
abscissa (int k, double xmin, double xmax, int xpoints)
{
	#pragma omp critical
	{
		if (IndexDictionary.find(xpoints) == IndexDictionary.end()) {calculate_weights(xpoints);}
	}
	return points[IndexDictionary[xpoints]](k);
}

template<>
inline double Quadrator<GAUSS_HERMITE>::
weight (int k, int xpoints)
{
	if (xpoints==1) {return M_SQRTPI;}
	#pragma omp critical
	{
		if (IndexDictionary.find(xpoints) == IndexDictionary.end()) {calculate_weights(xpoints);}
	}
	return exp( pow( points[IndexDictionary[xpoints]](k),2 ) ) * weights[IndexDictionary[xpoints]](k);
}

// for compatibility
template<>
inline double Quadrator<GAUSS_HERMITE>::
weight (int k, double xmin, double xmax, int xpoints)
{
	return weight(k,xpoints); 
}

template<>
double Quadrator<GAUSS_HERMITE>::
infigrate (double (*f)(double), int xpoints)
{
	double res = 0.;
	#ifndef QUADRATOR_DONT_USE_OPENMP
	#pragma omp parallel for reduction(+:res)
	#endif
	for (int ix=0; ix<xpoints; ++ix)
	{
		res += weight(ix,xpoints) * f(abscissa(ix,xpoints));
	}
	return res;
}
//-----------------------------</Hermite-Gauss>---------------------------

////-----------------------------<Clenshaw-Curtis>---------------------------

template<>
void Quadrator<CLENSHAW_CURTIS>::
calculate_weights (int xpoints)
{
	int Nfft = (xpoints-1)/2+1;
	
//	weights.resize(Nfft);
	Eigen::VectorXd curr_weights(Nfft);
	for (int k=0; k<Nfft; ++k)
	{
		curr_weights(k) = 2.*pow(xpoints-1.,-1)*pow(1.-4.*pow(k,2),-1);
	}
	
//	#ifndef QUADRATOR_DONT_USE_FFTWOMP
//	int fftw_init_threads(void);
//	fftw_plan_with_nthreads(omp_get_max_threads());
//	#endif
//	fftw_plan plan = fftw_plan_r2r_1d(Nfft, curr_weights.data(),curr_weights.data(), FFTW_REDFT00,FFTW_ESTIMATE);
//	fftw_execute(plan);
//	fftw_destroy_plan(plan);
//	#ifndef QUADRATOR_DONT_USE_FFTWOMP
//	void fftw_cleanup_threads(void);
//	#endif
	
	curr_weights(0) = 1./(xpoints*(xpoints-2.));
	
	IndexDictionary.insert(std::pair<int,int>(xpoints,IndexDictionary.size()));
	weights.push_back(curr_weights);
}

template<>
inline double Quadrator<CLENSHAW_CURTIS>::
abscissa (int k, int xpoints)
{
	return ChebyshevLobattoAbscissa(k,xpoints);
}

template<>
inline double Quadrator<CLENSHAW_CURTIS>::
abscissa (int k, double xmin, double xmax, int xpoints)
{
	return ChebyshevLobattoAbscissa(k,xmin,xmax,xpoints);
}

template<>
inline double Quadrator<CLENSHAW_CURTIS>::
weight (int k, int xpoints)
{
	if (xpoints==1) {return 2.;}
	int Nfft = (xpoints-1)/2+1;
//	if (weights.rows()!=Nfft) {calculate_weights(xpoints);}
	#pragma omp critical
	{
		if (IndexDictionary.find(xpoints) == IndexDictionary.end()) {calculate_weights(xpoints);}
	}
//	if (k>=Nfft) {return weights(xpoints-k-1);}
//	else         {return weights(k);}
	if (k>=Nfft) {return weights[IndexDictionary[xpoints]](xpoints-k-1);}
	else         {return weights[IndexDictionary[xpoints]](k);}
}

template<>
inline double Quadrator<CLENSHAW_CURTIS>::
weight (int k, double xmin, double xmax, int xpoints)
{
	if (xpoints==1) {return xmax-xmin;}
	int Nfft = (xpoints-1)/2+1;
//	if (weights.rows()!=Nfft) {calculate_weights(xpoints);}
	#pragma omp critical
	{
		if (IndexDictionary.find(xpoints) == IndexDictionary.end()) {calculate_weights(xpoints);}
	}
	double a = (xmax-xmin)*0.5;
//	if (k>=Nfft) {return a*weights(xpoints-k-1);}
//	else         {return a*weights(k);}
	if (k>=Nfft) {return a*weights[IndexDictionary[xpoints]](xpoints-k-1);}
	else         {return a*weights[IndexDictionary[xpoints]](k);}
}
////-----------------------------</Clenshaw-Curtis>---------------------------

//-----------------------------<Newton-Cotes closed>---------------------------
template<>
inline double Quadrator<NEWTON_COTES_CLOSED>::
abscissa (int k, double xmin, double xmax, int xpoints)
{
	if (xpoints==1) {return 0.5*(xmax-xmin);}
	double step = (xmax-xmin)/(xpoints-1);
	return xmin + k*step;
}
//-----------------------------</Newton-Cotes closed>---------------------------

//-----------------------------<Newton-Cotes open>---------------------------
template<>
inline double Quadrator<NEWTON_COTES_OPEN>::
abscissa (int k, double xmin, double xmax, int xpoints)
{
	if (xpoints==1) {return 0.5*(xmax-xmin);}
	double step = (xmax-xmin)/(xpoints+1);
	return xmin + (k+1)*step;
}
//-----------------------------</Newton-Cotes open>---------------------------

#endif

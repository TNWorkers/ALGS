#ifndef ORTHPOLYBASE
#define ORTHPOLYBASE

#include <list>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/FFT>
using namespace Eigen;

#include "LanczosSolver.h"
#include "OrthPolyEval.h"

enum KERNEL_CHOICE {NOKERNEL=0, JACKSON=1, LORENTZ=2};

int size (const double &x)
{
	return 1;
}

double at (const double &x, int i)
{
	return x;
}

int size (const ArrayXd &V)
{
	return V.rows();
}

double at (const ArrayXd &V, int i)
{
	return V(i);
}

template<typename Hamiltonian, typename VectorType>
class OrthPolyBase
{
public:
	
	OrthPolyBase(){};
	OrthPolyBase (const Hamiltonian &H, double padding_input=0.005);
	OrthPolyBase (double Emin_input, double Emax_input,  double padding_input=0.005);
	
	inline double get_Emin() const {return Emin;}
	inline double get_Emax() const {return Emax;}
	inline double get_a() const {return a;}
	inline double get_b() const {return b;}
	
	string baseinfo (string label="OrthPolyBase") const;
	int mvms() const {return N_mvm;};
	
	void set_scalings (double Emin_input, double Emax_input, double padding_input);
	
	static double kernel (int n, int N, KERNEL_CHOICE K);
	
protected:
	
	double Emin, Emax;
	double a, b, alpha, beta;
	double padding = 0.005;
	
	VectorXd  fct (const VectorXd &moments, int Npoints, bool REVERSE=false, KERNEL_CHOICE KERNEL_input=JACKSON);
	VectorXcd fft (const VectorXd &moments, int Npoints, KERNEL_CHOICE KERNEL_input=JACKSON);
	
//	double h (int ix, int iy, int iz);
//	
//	void calc_first (const Hamiltonian &H, const VectorType &V0, VectorType &V1);
//	void calc_next (const Hamiltonian &H, VectorType &V0, VectorType &V1);
//	void calc_next (const Hamiltonian &H, const VectorType &V0, const VectorType &V1, VectorType &Vnext);
	
	int N_mvm;
	size_t dimH;
};

template<typename Hamiltonian, typename VectorType>
OrthPolyBase<Hamiltonian,VectorType>::
OrthPolyBase(double Emin_input, double Emax_input, double padding_input)
{
	set_scalings(Emin_input,Emax_input,padding_input);
}

template<typename Hamiltonian, typename VectorType>
OrthPolyBase<Hamiltonian,VectorType>::
OrthPolyBase (const Hamiltonian &H, double padding_input)
{
	LanczosSolver<Hamiltonian,VectorType,double> Lutz;
	LanczosSolver<Hamiltonian,VectorType,double> Lucy;
	
	if (dim(H) < LANCZOS_MEMORY_THRESHOLD/2)
	#ifndef ORTHPOLYBASE_DONT_USE_OPENMP
	#pragma omp parallel sections
	#endif
	{
		#ifndef ORTHPOLYBASE_DONT_USE_OPENMP
		#pragma omp section
		#endif
		{
			#pragma omp critical
			{
				#ifdef _OPENMP
//				lout << "Emin, thread: " << omp_get_thread_num() << endl;
				#endif
			}
			Emin = Lutz.Emin(H);
//			lout << Lutz.info() << endl;
		}
		#ifndef ORTHPOLYBASE_DONT_USE_OPENMP
		#pragma omp section
		#endif
		{
			#pragma omp critical
			{
				#ifdef _OPENMP
//				lout << "Emax, thread: " << omp_get_thread_num() << endl;
				#endif
			}
			Emax = Lucy.Emax(H);
//			lout << Lucy.info() << endl;
		}
	}
	else
	{
		Emin = Lutz.Emin(H);
		Emax = Lucy.Emax(H);
	}
	
	set_scalings(Emin,Emax,padding_input);
}

template<typename Hamiltonian, typename VectorType>
string OrthPolyBase<Hamiltonian,VectorType>::
baseinfo (string label) const
{
	stringstream ss;
	ss << label << ":" 
	   << " Emin=" << Emin 
	   << ", Emax=" << Emax
	   << ", ½width a=" << a
	   << ", centre b=" << b
	   << ", padding=" << padding;
	return ss.str();
}

template<typename Hamiltonian, typename VectorType>
inline double OrthPolyBase<Hamiltonian,VectorType>::
kernel (int n, int N, KERNEL_CHOICE KERNEL_input)
{
	double out = 0.;
	
	if (KERNEL_input == JACKSON)
	{
		double k = M_PI/(N+1.);
		out = ((N-n+1.)*cos(n*k)+sin(n*k)/tan(k))/(N+1.);
	}
	else if (KERNEL_input == LORENTZ)
	{
		out = sinh(4.*(1.-static_cast<double>(n)/N))/sinh(4.);
	}
	else if (KERNEL_input == NOKERNEL)
	{
		out = 1.;
	}
	return out;
}

template<typename Hamiltonian, typename VectorType>
void OrthPolyBase<Hamiltonian,VectorType>::
set_scalings (double Emin_input, double Emax_input, double padding_input)
{
	N_mvm = 0;
	
	Emin = Emin_input;
	Emax = Emax_input;
	padding = padding_input;
	
	if (Emin == Emax)
	{
		a = 1.;
		b = Emax+1e-10; // offset required for DMRG, otherwise failure in H^2 for product states
	}
	else
	{
//		a = (Emax-Emin)/(2.-0.01)+1e-10;
//		a = (Emax-Emin)*0.5/(1.-padding)+1e-10;
		a = (Emax-Emin)*0.5/(1.-padding);
		b = (Emax+Emin)*0.5;
	}
	alpha = 1./a;
	beta  = -b/a;
}

//template<typename Hamiltonian, typename VectorType>
//inline double OrthPolyBase<Hamiltonian,VectorType>::
//h (int ix, int iy, int iz)
//{
//	double res = 1.;
//	if (ix>0) {res *= 2;}
//	if (iy>0) {res *= 2;}
//	if (iz>0) {res *= 2;}
//	return res;
//}

//template<typename Hamiltonian, typename VectorType>
//void OrthPolyBase<Hamiltonian,VectorType>::
//calc_first (const Hamiltonian &H, const VectorType &V0, VectorType &V1)
//{
//	HxV(H, V0,V1); ++N_mvm; // V1 = H*V0;
//	V1 *= alpha; // V1 = α·H*V0;
//	V1 += beta * V0; // V1 = α·H*V0 + β·V0;
//}

//template<typename Hamiltonian, typename VectorType>
//void OrthPolyBase<Hamiltonian,VectorType>::
//calc_next (const Hamiltonian &H, VectorType &V0, VectorType &V1)
//{
//	VectorType Vtmp = V0;
//	HxV(H, V1,V0); ++N_mvm; // V0 = H*V1
//	V0 *= 2.*alpha; // V0 = 2·α·H*V1
//	V0 += 2.*beta * V1; // V0 = 2·α·H*V1 + 2·β·V1;
//	V0 -= Vtmp; // V0 = 2·α·H*V1 + 2·β·V1 - V0;
//	swap(V0,V1);
//}

//template<typename Hamiltonian, typename VectorType>
//void OrthPolyBase<Hamiltonian,VectorType>::
//calc_next (const Hamiltonian &H, const VectorType &V0, const VectorType &V1, VectorType &Vnext)
//{
//	HxV(H, V1,Vnext); ++N_mvm; // Vnext = H*V1
//	Vnext = 2.*alpha*Vnext + 2.*beta*V1 - V0; // Vnext = 2·α·H*V1 + 2·β·V1 - V0;
//}

template<typename Hamiltonian, typename VectorType>
VectorXd OrthPolyBase<Hamiltonian,VectorType>::
fct (const VectorXd &moments, int Npoints, bool REVERSE, KERNEL_CHOICE KERNEL_input)
{
	// using Eigen's FFT
	VectorXcd lambda(Npoints);
	lambda(0) = moments(0) * kernel(0,moments.rows(),KERNEL_input);
	for (int n=1; n<moments.rows(); ++n)
	{
		double phase = (REVERSE==true)? 1. : pow(-1.,n); // when reversing, phases of (-1)^n cancel out
//		lambda(n) = 2.*moments(n) * kernel(n,moments.rows(),KERNEL_input) * exp(complex<double>(0,-M_PI_2*n/Npoints))*pow(-1.,n);
		lambda(n) = 2.*moments(n) * kernel(n,moments.rows(),KERNEL_input) * exp(complex<double>(0,-M_PI_2*n/Npoints)) * phase;
	}
	lambda.segment(moments.rows(),Npoints-moments.rows()).setZero();
	
	Eigen::FFT<double> FourierTransformer;
	VectorXcd flambda(Npoints);
	FourierTransformer.fwd(flambda,lambda);
	
	VectorXd Vout(Npoints);
	for (int j=0; j<Npoints/2; ++j)
	{
		Vout(2*j)   = flambda(j).real();
		Vout(2*j+1) = flambda(Npoints-1-j).real();
	}
	return Vout;

	// explicitly
//	cout << "FCT:" << endl;
//	VectorXd Vout(Npoints);
//	for (int k=0; k<Npoints; ++k)
//	{
////		cout << "k=" << k << " " << (1.-0.15)*cos(M_PI*(k+0.5)/Npoints+M_PI) << endl;
//		Vout(k) = moments(0) * kernel(0,moments.rows(),KERNEL_input);
//		for (int n=1; n<moments.rows(); ++n)
//		{
//			double phase = (REVERSE==true)? 1. : pow(-1.,n); // when reversing, phases of (-1)^n cancel out
//			Vout(k) += 2.*moments(n) * kernel(n,moments.rows(),KERNEL_input) * cos(M_PI*n*(k+0.5)/Npoints) * phase;
//		}
//	}
//	return Vout;

//	// using FFTW
//	VectorXd Vout = moments;
//	for (int n=0; n<moments.rows(); ++n)
//	{
//		double phase = (REVERSE==true)? 1. : pow(-1.,n);
//		Vout(n) *= kernel(n,moments.rows(),KERNEL_input) * phase;
//	}
//	fftw_plan plan = fftw_plan_r2r_1d(moments.rows(), Vout.data(),Vout.data(), FFTW_REDFT01,FFTW_ESTIMATE);
//	fftw_execute(plan);
//	fftw_destroy_plan(plan);
//	return Vout;

//	// using FFTW
//	VectorXd Vout = ODOSmoments;
//	for (int n=0; n<N_ODOSmoments; ++n)
//	{
//		Vout(n) *= JacksonKernel(n,N_ODOSmoments) * pow(-1,n);
//	}
//	fftw_plan plan = fftw_plan_r2r_1d(N_ODOSmoments, Vout.data(),Vout.data(), FFTW_REDFT01,FFTW_ESTIMATE);
//	fftw_execute(plan);
//	fftw_destroy_plan(plan);
//	return Vout;
}

template<typename Hamiltonian, typename VectorType>
VectorXcd OrthPolyBase<Hamiltonian,VectorType>::
fft (const VectorXd &moments, int Npoints, KERNEL_CHOICE KERNEL_input)
{
	// using Eigen's FFT
	VectorXcd lambda(Npoints);
	lambda(0) = moments(0)*kernel(0,moments.rows(),KERNEL_input);
	for (int n=1; n<moments.rows(); ++n)
	{
		lambda(n) = 2.*moments(n) * kernel(n,moments.rows(),KERNEL_input) * exp(complex<double>(0,-M_PI_2*n/Npoints))*pow(-1.,n);
	}
	lambda.segment(moments.rows(),Npoints-moments.rows()).setZero();
	
	Eigen::FFT<double> FourierTransformer;
	VectorXcd flambda(Npoints);
	FourierTransformer.fwd(flambda,lambda);
	
	VectorXcd Vout(Npoints);
	for (int j=0; j<Npoints/2; ++j)
	{
		// check this:
		Vout(2*j)   = flambda(j);
		Vout(2*j+1) = conj(flambda(Npoints-1-j));
	}
	return -complex<double>(0,-M_PI) * Vout;
}

#endif

#ifndef CHEBYSHEVTRANSFORMER
#define CHEBYSHEVTRANSFORMER

#include <Eigen/Dense>
#include <Eigen/Sparse>
/*#include <armadillo>*/

//#ifndef CHEBTRANS_DONT_USE_FFTWOMP
//#include <fftw3.h> // compile with -lfftw3 -lfftw3_omp
//#endif
#include "IntervalIterator.h"
#include "ChebyshevAbscissa.h"
#include "Quadrator.h"
#include "Stopwatch.h"
//#include "InterpolGSL.h"
#include "gsl/gsl_sf_gamma.h"
//#include "trigfactor.h"
#include "Stopwatch.h"

double ChebyshevT (int n, double x)
{
	if      (n==0) {return 1.;}
	else if (n==1) {return x;}
	else if (n==2) {return 2.*pow(x,2)-1.;}
	else if (n==3) {return 4.*pow(x,3)-3.*x;}
//	else if (n==4) {return 8.*pow(x,4)-8.*x*x+1.;}
//	else if (n==5) {return 16.*pow(x,5)-20.*pow(x,3)+5.*x;}
//	else if (n==6) {return 32.*pow(x,6)-48.*pow(x,4)+18.*pow(x,2)-1.;}
//	else if (n==7) {return 64.*pow(x,7)-112.*pow(x,5)+56.*pow(x,3)-7.*x;}
//	else if (n==8) {return 128.*pow(x,8)-256.*pow(x,6)+160.*pow(x,4)-32.*pow(x,2)+1.;}
//	else if (n==9) {return 256.*pow(x,9)-576.*pow(x,7)+432.*pow(x,5)-120.*pow(x,3)+9.*x;}
	
	double tnm1 = 4.*pow(x,3)-3.*x;
	double tnm2 = 2.*pow(x,2)-1.;
	double tn = tnm1;
	
	for (int l=4; l<=n; ++l)
	{
		tn = 2.*x*tnm1-tnm2;
		tnm2 = tnm1;
		tnm1 = tn;
	}
	return tn;
}

enum ORTHO_POLY {pLEGENDRE, pLEGENDRE_SHIFTED, pCHEBYSHEV1, pCHEBYSHEV2, pCHEBYSHEV3, pCHEBYSHEV4, pGEGENBAUER};

class ChebyshevTransformer
{
public:
	
	ChebyshevTransformer (double xmin_input=-1., double xmax_input=+1.);
	ChebyshevTransformer (double (*f_input)(double), double xmin_input=-1., double xmax_input=1.);
	
	ChebyshevTransformer (double xmin_input, double xmax_input, double ymin_input, double ymax_input);
//	ChebyshevTransformer (double (*f2d_input)(double,double), double xmin_input=-1., double xmax_input=1., double ymin_input=-1., double ymax_input=1.);
	
	ChebyshevTransformer (double xmin_input, double xmax_input, double ymin_input, double ymax_input, double zmin_input, double zmax_input);
//	ChebyshevTransformer (double (*f3d_input)(double,double,double), double xmin_input=-1., double xmax_input=1., double ymin_input=-1., double ymax_input=1., double zmin_input=-1., double zmax_input=1.);
	
//	inline Eigen::VectorXd moments()   {return stored_moments;}; // via Gauss-Chebyshev with FCT
//	inline Eigen::MatrixXd moments2d() {return stored_moments2d;};
//	inline arma::cube      moments3d() {return stored_moments3d;};
	
	void compute_recursionCoefficients_modChebyshev (int n); // calculates the recursion coefficients alpha & beta for weight f(x) via the modified Chebyshev algorithm (Gautschi 1982, 2004)
	template<QUAD_RULE qrule> 
	void compute_recursionCoefficients_GanderKarp (int n, double err=1e-6); // calculates the recursion coefficients alpha & beta for weight f(x) via the Gander-Karp algorithm
//	void compute_recursionCoefficients_matrix (int n);
	void compute_analyticalRecursion (ORTHO_POLY OP, int n, double p1=0.); // some analytically known recursion coefficients with optional parameter p1
	inline Eigen::VectorXd alpha() {return stored_alpha;}
	inline Eigen::VectorXd beta()  {return stored_beta;}
	
//	double reconstruct (double x);
//	double reconstruct (double x, double y);
//	double reconstruct (double x, double y, double z);
	
/*	void inject (const arma::cube &moments3d);*/
	
private:
	
	double (*f)(double x);
	double xmin, xmax;
	double a, b;
	
//	double (*f2d)(double x, double y);
	double ymin, ymax;
	double ay, by;
	Eigen::MatrixXd stored_moments2d;
	
//	double (*f3d)(double x, double y, double z);
	double zmin, zmax;
	double az, bz;
/*	arma::cube stored_moments3d;*/
	
	int N_moments;
	Eigen::VectorXd stored_moments;
	
	void compute_moments_by_fct1dim(); // calculates the first N_moments=2^20 Chebyshev moments of f(x)
//	void compute_moments_by_fct2dim();
//	void compute_moments_by_fct3dim();
	double cheb_beta (int n); // monic Chebyshev recursion coefficients: T_n+1(x) = x*T_n(x) - beta(n)*T_n-1(x)
	
	void make_monic(); // in-place rescaling to make moments calculated from monic Chebyshev
	void undo_monic(); // undoes the above
	
	Eigen::VectorXd stored_alpha;
	Eigen::VectorXd stored_beta;
	
//	void make_ChebInterp (int order);
//	vector<Interpol<GSL> > ChebInterpol;
	bool GOT_CHEBINTERPOL;
};

ChebyshevTransformer::
ChebyshevTransformer (double xmin_input, double xmax_input)
:xmin(xmin_input), xmax(xmax_input), GOT_CHEBINTERPOL(false)
{
	a = 0.5*(xmax-xmin);
	b = 0.5*(xmin+xmax);
}

ChebyshevTransformer::
ChebyshevTransformer (double xmin_input, double xmax_input, double ymin_input, double ymax_input)
:xmin(xmin_input), xmax(xmax_input), ymin(ymin_input), ymax(ymax_input), GOT_CHEBINTERPOL(false)
{
	a = 0.5*(xmax-xmin);
	b = 0.5*(xmin+xmax);
	ay = 0.5*(ymax-ymin);
	by = 0.5*(ymin+ymax);
}

ChebyshevTransformer::
ChebyshevTransformer (double xmin_input, double xmax_input, double ymin_input, double ymax_input, double zmin_input, double zmax_input)
:xmin(xmin_input), xmax(xmax_input), ymin(ymin_input), ymax(ymax_input), zmin(zmin_input), zmax(zmax_input), GOT_CHEBINTERPOL(false)
{
	a = 0.5*(xmax-xmin);
	b = 0.5*(xmin+xmax);
	ay = 0.5*(ymax-ymin);
	by = 0.5*(ymin+ymax);
	az = 0.5*(zmax-zmin);
	bz = 0.5*(zmin+zmax);
}

ChebyshevTransformer::
ChebyshevTransformer (double (*f_input)(double), double xmin_input, double xmax_input)
:f(f_input), xmin(xmin_input), xmax(xmax_input), GOT_CHEBINTERPOL(false)
{
	N_moments = pow(2,17);
	stored_moments.resize(N_moments);
	stored_moments.setZero();
	
	a = 0.5*(xmax-xmin);
	b = 0.5*(xmin+xmax);
	
	compute_moments_by_fct1dim();
}

void ChebyshevTransformer::
compute_moments_by_fct1dim()
{
	//Stopwatch Chronos;
	IntervalIterator xit(xmin,xmax,N_moments,ChebyshevAbscissa);
	
	for (xit=xit.begin(); xit<xit.end(); ++xit)
	{
		double x_scaled = (xit.value()-b)/a;
		stored_moments(xit.index()) = f(xit.value()) * sqrt(1.-x_scaled*x_scaled);
//		cout << "root: " << sqrt(1.-x_scaled*x_scaled) << endl;
	}
	
//	// using FFTW
//	#ifndef CHEBTRANS_DONT_USE_FFTWOMP
//	int fftw_init_threads(void);
//	fftw_plan_with_nthreads(omp_get_max_threads());
//	#endif
//	fftw_plan p = fftw_plan_r2r_1d(N_moments, stored_moments.data(),stored_moments.data(), FFTW_REDFT10,FFTW_ESTIMATE);
//	fftw_execute(p);
//	fftw_destroy_plan(p);
//	#ifndef CHEBTRANS_DONT_USE_FFTWOMP
//	void fftw_cleanup_threads(void);
//	#endif
	
//	// explicit:
//	VectorXd fct(N_moments); fct.setZero();
//	for (int k=0; k<N_moments; ++k)
//	for (int j=0; j<N_moments; ++j)
//	{
//		fct(k) += 2. * stored_moments(j) * cos(M_PI*(j+0.5)*k/N_moments);
//	}
	
	// using FFT
	Eigen::FFT<double> fft;
	VectorXd lambda(2*N_moments); lambda.setZero();
	lambda.head(N_moments) = stored_moments;
	VectorXcd flambda(2*N_moments);
	fft.fwd(flambda,lambda);
	for (int k=0; k<N_moments; ++k) flambda(k) *= 2. * exp(-1.i*M_PI*double(k)/double(2.*N_moments));
	VectorXd fct = flambda.head(N_moments).real();
	
	stored_moments = fct;
	for (int n=1; n<N_moments; n+=2) {stored_moments(n) *= -1.;}
	stored_moments *= M_PI_2/N_moments;
	//Chronos.check("moments");
}

//ChebyshevTransformer::
//ChebyshevTransformer (double (*f2d_input)(double,double), double xmin_input, double xmax_input, double ymin_input, double ymax_input)
//:f2d(f2d_input), xmin(xmin_input), xmax(xmax_input), ymin(ymin_input), ymax(ymax_input), GOT_CHEBINTERPOL(false)
//{
//	N_moments = pow(2,10);
//	stored_moments2d.resize(N_moments,N_moments);
//	stored_moments2d.setZero();
//	
//	a = 0.5*(xmax-xmin);
//	b = 0.5*(xmin+xmax);
//	ay = 0.5*(ymax-ymin);
//	by = 0.5*(ymin+ymax);
//	
//	compute_moments_by_fct2dim();
//}

//void ChebyshevTransformer::
//compute_moments_by_fct2dim()
//{
//	IntervalIterator xit(xmin,xmax,N_moments,ChebyshevAbscissa);
//	IntervalIterator yit(ymin,ymax,N_moments,ChebyshevAbscissa);
//	
//	for (xit=xit.begin(); xit<xit.end(); ++xit)
//	for (yit=yit.begin(); yit<yit.end(); ++yit)
//	{
//		double x_scaled = (xit.value()-b)/a;
//		double y_scaled = (yit.value()-by)/ay;
//		stored_moments2d(xit.index(),yit.index()) = f2d(xit.value(),yit.value()) * sqrt( (1.-pow(x_scaled,2))*(1.-pow(y_scaled,2)) );
//	}
//	
//	#ifndef CHEBTRANS_DONT_USE_FFTWOMP
//	int fftw_init_threads(void);
//	fftw_plan_with_nthreads(omp_get_max_threads());
//	#endif
//	fftw_plan p = fftw_plan_r2r_2d(N_moments,N_moments, 
//					stored_moments2d.data(), stored_moments2d.data(), 
//					FFTW_REDFT10,FFTW_REDFT10,FFTW_ESTIMATE);
//	fftw_execute(p);
//	fftw_destroy_plan(p);
//	#ifndef CHEBTRANS_DONT_USE_FFTWOMP
//	void fftw_cleanup_threads(void);
//	#endif
//	
//	for (int n=0; n<N_moments; ++n)
//	for (int m=0; m<N_moments; ++m)
//	{
//		stored_moments2d(n,m) *= pow(-1.,n+m);
//	}
//	stored_moments2d *= pow(M_PI_2/N_moments,2.);
//	
//	// reconstruction of the function with:
//	//sum_ij CT.moments2d()(i,j) * cheb(i,xval) * cheb(j,yval) * h(i,j) * pow(M_1_PI,2.)/sqrt(1.-xval*xval)/sqrt(1.-yval*yval);
////	inline double h (int ix, int iy)
////	{
////		double res=1.;
////		if (ix>0) {res *= 2.;}
////		if (iy>0) {res *= 2.;}
////		return res;
////	}
//}

//ChebyshevTransformer::
//ChebyshevTransformer (double (*f3d_input)(double,double,double), double xmin_input, double xmax_input, double ymin_input, double ymax_input, double zmin_input, double zmax_input)
//:f3d(f3d_input), xmin(xmin_input), xmax(xmax_input), ymin(ymin_input), ymax(ymax_input), zmin(zmin_input), zmax(zmax_input), GOT_CHEBINTERPOL(false)
//{
//	N_moments = pow(2,7);
///*	stored_moments3d = arma::zeros(N_moments,N_moments,N_moments);*/
//	
//	a = 0.5*(xmax-xmin);
//	b = 0.5*(xmin+xmax);
//	ay = 0.5*(ymax-ymin);
//	by = 0.5*(ymin+ymax);
//	az = 0.5*(zmax-zmin);
//	bz = 0.5*(zmin+zmax);
//	
//	compute_moments_by_fct3dim();
//}

//void ChebyshevTransformer::
//compute_moments_by_fct3dim()
//{
//	IntervalIterator xit(xmin,xmax,N_moments,ChebyshevAbscissa);
//	IntervalIterator yit(ymin,ymax,N_moments,ChebyshevAbscissa);
//	IntervalIterator zit(zmin,zmax,N_moments,ChebyshevAbscissa);
//	
//	for (xit=xit.begin(); xit<xit.end(); ++xit)
//	for (yit=yit.begin(); yit<yit.end(); ++yit)
//	for (zit=zit.begin(); zit<zit.end(); ++zit)
//	{
//		double x_scaled = (xit.value()-b)/a;
//		double y_scaled = (yit.value()-by)/ay;
//		double z_scaled = (zit.value()-bz)/az;
//		stored_moments3d(xit.index(),yit.index(),zit.index()) = f3d(xit.value(),yit.value(),zit.value()) 
//			* sqrt( (1.-pow(x_scaled,2))*(1.-pow(y_scaled,2))*(1.-pow(z_scaled,2)) );
//	}
//	
//	#ifndef CHEBTRANS_DONT_USE_FFTWOMP
//	int fftw_init_threads(void);
//	fftw_plan_with_nthreads(omp_get_max_threads());
//	#endif
//	fftw_plan p = fftw_plan_r2r_3d(N_moments,N_moments,N_moments, 
//					stored_moments3d.memptr(), stored_moments3d.memptr(), 
//					FFTW_REDFT10,FFTW_REDFT10,FFTW_REDFT10,FFTW_ESTIMATE);
//	fftw_execute(p);
//	fftw_destroy_plan(p);
//	#ifndef CHEBTRANS_DONT_USE_FFTWOMP
//	void fftw_cleanup_threads(void);
//	#endif
//	
//	for (int n=0; n<N_moments; ++n)
//	for (int m=0; m<N_moments; ++m)
//	for (int l=0; l<N_moments; ++l)
//	{
//		stored_moments3d(n,m,l) *= pow(-1.,n+m+l);
//	}
//	stored_moments3d *= pow(M_PI_2/N_moments,3.);
//}

//void ChebyshevTransformer::
//inject (const arma::cube &moments3d)
//{
//	assert(moments3d.n_rows==moments3d.n_cols && moments3d.n_rows==moments3d.n_slices);
//	stored_moments3d = moments3d;
//	N_moments = moments3d.n_rows;
//}

//double ChebyshevTransformer::
//reconstruct (double x)
//{
//	if (GOT_CHEBINTERPOL==false) {make_ChebInterp(1024);}

//	double x_scaled = (x-b)/a;
//	
//	double res = 0;
//	#pragma omp parallel for reduction(+:res)
//	for (int ix=0; ix<1024; ++ix)
//	{
//		res += stored_moments(ix) * 
////				ChebyshevT(ix,x_scaled)*ChebyshevT(iy,y_scaled) * 
//				ChebInterpol[ix].evaluate(x_scaled) *
//				trigfactor({ix}) / sqrt(1.-pow(x_scaled,2));
//	}
//	res *= M_1_PI;
//	return res;
//}

//double ChebyshevTransformer::
//reconstruct (double x, double y)
//{
//	if (GOT_CHEBINTERPOL==false) {make_ChebInterp(N_moments);}

//	double x_scaled = (x-b)/a;
//	double y_scaled = (y-by)/ay;
//	
//	double res = 0;
//	#pragma omp parallel for collapse(2) reduction(+:res)
//	for (int ix=0; ix<N_moments; ++ix)
//	for (int iy=0; iy<N_moments; ++iy)
//	{
//		res += stored_moments2d(ix,iy) * 
////				ChebyshevT(ix,x_scaled)*ChebyshevT(iy,y_scaled) * 
//				ChebInterpol[ix].evaluate(x_scaled) * ChebInterpol[iy].evaluate(y_scaled) *
//				trigfactor({ix,iy}) / sqrt( (1.-pow(x_scaled,2))*(1.-pow(y_scaled,2)) );
////				trigfactor({ix,iy}) / sqrt( (a-pow(x-b,2))*(ay-pow(y-by,2)) );
//	}
//	res *= pow(M_1_PI,2);
//	return res;
//}

//double ChebyshevTransformer::
//reconstruct (double x, double y, double z)
//{
//	if (GOT_CHEBINTERPOL==false) {make_ChebInterp(N_moments);}
//	
//	double x_scaled = (x-b)/a;
//	double y_scaled = (y-by)/ay;
//	double z_scaled = (z-bz)/az;
//	
//	double res = 0;
//	#pragma omp parallel for collapse(3) reduction(+:res)
//	for (int ix=0; ix<N_moments; ++ix)
//	for (int iy=0; iy<N_moments; ++iy)
//	for (int iz=0; iz<N_moments; ++iz)
//	{
//		res += stored_moments3d(ix,iy,iz) * 
////				ChebyshevT(ix,x_scaled) * ChebyshevT(iy,y_scaled) * ChebyshevT(iz,z_scaled) * 
//				ChebInterpol[ix].evaluate(x_scaled) * ChebInterpol[iy].evaluate(y_scaled) * ChebInterpol[iz].evaluate(z_scaled) *
//				trigfactor({ix,iy,iz}) / sqrt( (1.-pow(x_scaled,2))*(1.-pow(y_scaled,2))*(1.-pow(z_scaled,2)) );
////				trigfactor({ix,iy,iz}) / sqrt( (a-pow(x-b,2))*(ay-pow(y-by,2))*(az-pow(z-bz,2)) );
//	}
//	res *= pow(M_1_PI,3);
//	return res;
//}

//void ChebyshevTransformer::
//make_ChebInterp (int max_order)
//{
//	ChebInterpol.reserve(max_order);
//	
//	for (int n=0; n<max_order; ++n)
//	{
//		int Ninterpoints = 10000;
//		IntervalIterator xit(-1.,1.,Ninterpoints);
//		
//		Interpol<GSL> IP(-1.,1.,Ninterpoints);
//		ChebInterpol.push_back(IP);
//		for (xit=xit.begin(); xit<xit.end(); ++xit)
//		{
//			ChebInterpol[ChebInterpol.size()-1].insert(xit.index(), ChebyshevT(n,*xit));
//		}
//		ChebInterpol[ChebInterpol.size()-1].set_splines();
//		
////		cout << ChebInterpol[ChebInterpol.size()-1].evaluate(-1.) << "\t" << ChebInterpol[ChebInterpol.size()-1].evaluate(1.) << endl;
//	}
//	GOT_CHEBINTERPOL = true;
//}

inline void ChebyshevTransformer::
make_monic()
{
	for (int n=2; n<N_moments; ++n)
	{
		stored_moments(n) /= pow(2.,n-1);
	}
};

inline void ChebyshevTransformer::
undo_monic()
{
	for (int n=2; n<N_moments; ++n)
	{
		stored_moments(n) *= pow(2.,n-1);
	}
};

inline double ChebyshevTransformer::
cheb_beta (int n)
{
	if      (n==0) {return M_PI;}
	else if (n==1) {return 0.5;}
	else           {return 0.25;}
}

void ChebyshevTransformer::
compute_recursionCoefficients_modChebyshev (int n)
{
	assert(2*n<=N_moments);
	make_monic();
	stored_alpha.resize(n);
	stored_beta.resize(n);
	
	Eigen::VectorXd sigma0(2*n);
	Eigen::VectorXd sigma1(2*n);
	sigma0.setZero();
	sigma1.setZero();
	
	// init: k=0
	sigma0 = stored_moments.head(2*n); // sigma0=sigma(0,0:2n-1)=moments
	stored_alpha(0) = stored_moments(1)/stored_moments(0);
	stored_beta(0) = stored_moments(0);
	
	// init: k=1
	for (int l=1; l<2*n-1; ++l)
	{
		sigma1(l) = sigma0(l+1) - stored_alpha(0)*sigma0(l) + cheb_beta(l)*sigma0(l-1); // sigma1=sigma(1,:)
	}
	stored_alpha(1) = sigma1(2)/sigma1(1) - sigma0(1)/sigma0(0);
	stored_beta(1) = sigma1(1)/sigma0(0);
	
	// iteration: k>1
	for (int k=2; k<n; ++k)
	{
		for (int l=k; l<2*n-k; ++l)
		{
			sigma0(l) = sigma1(l+1) - stored_alpha(k-1)*sigma1(l) - stored_beta(k-1)*sigma0(l) + cheb_beta(l)*sigma1(l-1); //sigma0 = sigma(k,:)
		}
		sigma0.swap(sigma1); // sigma1=sigma(k,:), sigma0=sigma(k-1,:)
		stored_alpha(k) = sigma1(k+1)/sigma1(k) - sigma0(k)/sigma0(k-1);
		stored_beta(k) = sigma1(k)/sigma0(k-1);
	}
	
	if ((stored_beta.array()<0).any()) {cout << "ChebyshevTransformer: Instability in modified Chebyshev algorithm detected!" << endl;}
	
	undo_monic();
	
	// scaling
	stored_alpha = (a*stored_alpha.array()+b).matrix();
	stored_beta(0) *= a;
	stored_beta.tail(n-1) *= a*a;
}

template<QUAD_RULE qrule>
void ChebyshevTransformer::
compute_recursionCoefficients_GanderKarp (int n, double error_input)
{
	Eigen::Tridiagonalization<Eigen::MatrixXd> Tom;
	
	int Nmax = 4096;
	int Naux = n;
	stored_beta.resize(n); // memory used for error calculation
	stored_beta.setZero();
	double last_beta_err = 1.;
	
	Quadrator<qrule> Q;
	
	while (last_beta_err > error_input)
	{
		assert(Naux<=Nmax && "Matrix size for Gander-Karp tridiagonalization exceeded! Try reducing the accuracy, changing the quadrature rule (LEGENDRE?), reducing the amount of coefficients; or use the modified Chebyshev algorithm instead.");
		
		Eigen::SparseMatrix<double> A(Naux+1,Naux+1);
		vector<Eigen::Triplet<double> > coeff(3*Naux+1);
		coeff.push_back(Eigen::Triplet<double>(0,0, 1.));
		for (int i=0; i<Naux; ++i)
		{
			double x = Q.abscissa(i,xmin,xmax,Naux);
			assert(f(x)>0. && "Weight function has to be positive definite!");
			double w = sqrt( f(x) * Q.weight(i,xmin,xmax,Naux) );
			coeff.push_back(Eigen::Triplet<double>(i+1,i+1, x));
			coeff.push_back(Eigen::Triplet<double>(0,  i+1, w));
			coeff.push_back(Eigen::Triplet<double>(i+1,  0, w));
		}
		A.setFromTriplets(coeff.begin(),coeff.end());
		
//		// same with a dense matrix (seems to have the same performance)
//		Eigen::MatrixXd A(Naux+1,Naux+1);
//		A.setZero();
//		A(0,0) = 1.;
//		for (int i=0; i<Naux; ++i)
//		{
//			A(i+1,i+1) = Q.abscissa(i,xmin,xmax,Naux); // diagonal
//			A(0,i+1) = sqrt(f(Q.abscissa(i,xmin,xmax,Naux)) * Q.weight(i,xmin,xmax,Naux)); // 1st row
//			A(i+1,0) = A(0,i+1);  // 1st col = 1st row
//		}
		
		Tom.compute(A);
		
		last_beta_err = (stored_beta-Tom.subDiagonal().head(n)).cwiseAbs().maxCoeff();
		stored_beta = Tom.subDiagonal().head(n);
		Naux += n;
	}
	
	stored_alpha = Tom.diagonal().segment(1,n);
	stored_beta  = Tom.subDiagonal().head(n).cwiseAbs2();
}

//void ChebyshevTransformer::
//compute_recursionCoefficients_matrix (int n)
//{
//	assert(2*n<=N_moments);
//	make_monic();
//	stored_alpha.resize(n);
//	stored_beta.resize(n);
//	
//	Eigen::MatrixXd sigma(2*n,2*n);
//	sigma.setZero();
//	
//	sigma.row(0) = stored_moments.head(2*n).transpose();
//	stored_alpha(0) = stored_moments(1)/stored_moments(0);
//	stored_beta(0) = stored_moments(0);
//	
//	for (int l=1; l<2*n-1; ++l)
//	{
//		sigma(1,l) = sigma(0,l+1) - stored_alpha(0)*sigma(0,l) + cheb_beta(l)*sigma(0,l-1);
//	}
//	stored_alpha(1) = sigma(1,2)/sigma(1,1) - sigma(0,1)/sigma(0,0);
//	stored_beta(1) = sigma(1,1)/sigma(0,0);
//	
//	for (int k=2; k<n; ++k)
//	{
//		for (int l=k; l<2*n-k; ++l)
//		{
//			sigma(k,l) = sigma(k-1,l+1) - stored_alpha(k-1)*sigma(k-1,l) - stored_beta(k-1)*sigma(k-2,l) + cheb_beta(l)*sigma(k-1,l-1);
//		}
//		stored_alpha(k) = sigma(k,k+1)/sigma(k,k) - sigma(k-1,k)/sigma(k-1,k-1);
//		stored_beta(k) = sigma(k,k)/sigma(k-1,k-1);
//	}
//	
//	undo_monic();
//	
//	stored_alpha = (a*stored_alpha.array()+b).matrix();
//	stored_beta *= a*a;
//}

inline void ChebyshevTransformer::
compute_analyticalRecursion (ORTHO_POLY OP_input, int n, double p1)
{
	stored_alpha.resize(n);
	stored_beta.resize(n);
	
	// pLEGENDRE_SHIFTED with shifted borders -> go to Legendre
	if (OP_input==pLEGENDRE or (OP_input==pLEGENDRE_SHIFTED and xmin!=-1. and xmax!=1.))
	{
		stored_alpha.setZero();
		stored_beta(0) = 2.;
		for (int i=1; i<n; ++i) {stored_beta(i) = 1./(4.-pow(i,-2.));}
	}
	if (OP_input==pLEGENDRE_SHIFTED and xmin==-1. and xmax==-1.)
	{
		stored_alpha.setConstant(0.5);
		stored_beta(0) = 1.;
		for (int i=1; i<n; ++i) {stored_beta(i) = 0.25/(4.-pow(i,-2.));}
	}
	else if (OP_input==pCHEBYSHEV1)
	{
		stored_alpha.setZero();
		stored_beta.setConstant(0.25);
		stored_beta(0) = M_PI;
		stored_beta(1) = 0.5;
	}
	else if (OP_input==pCHEBYSHEV2)
	{
		stored_alpha.setZero();
		stored_beta.setConstant(0.25);
		stored_beta(0) = 0.5*M_PI;
	}
	else if (OP_input==pCHEBYSHEV3)
	{
		stored_alpha.setZero();
		stored_alpha(0) = 0.5;
		stored_beta.setConstant(0.25);
		stored_beta(0) = M_PI;
	}
	else if (OP_input==pCHEBYSHEV4)
	{
		stored_alpha.setZero();
		stored_alpha(0) = -0.5;
		stored_beta.setConstant(0.25);
		stored_beta(0) = M_PI;
	}
	else if (OP_input==pGEGENBAUER)
	{
		stored_alpha.setZero();
		stored_beta(0) = sqrt(M_PI)*gsl_sf_gamma(p1+0.5)/gsl_sf_gamma(p1+1.);
		for (int i=1; i<n; ++i)
		{
			stored_beta(i) = 0.25*i*(i+2.*p1-1.)/((i+p1)*(i+p1-1.));
		}
	}
	
	// scaling
	stored_alpha = (a*stored_alpha.array()+b).matrix();
	stored_beta *= a*a;
}

#endif

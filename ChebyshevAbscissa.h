#ifndef CHEBYSHEVABSCISSA
#define CHEBYSHEVABSCISSA

#include <gsl/gsl_math.h>
#include <Eigen/Dense>

//----------------<Chebyshev-Gauss>----------------
inline double ChebyshevAbscissa (int k, double xmin, double xmax, int xpoints)
{
	double a = (xmax-xmin)*0.5;
	double b = (xmax+xmin)*0.5;
	if (xpoints==1) {return b;}
	return -a*cos((k+0.5)*M_PI/xpoints)+b;
}

		// default: xmin=-1, xmax=+1
		inline double ChebyshevAbscissa (int k, int xpoints)
		{
			if (xpoints==1) {return 0;}
			return -cos((k+0.5)*M_PI/xpoints);
		}
		
inline double ChebyshevExtrema (int k, double xmin, double xmax, int xpoints)
{
	double a = (xmax-xmin)*0.5;
	double b = (xmax+xmin)*0.5;
	if (xpoints==1) {return b;}
	return -a*cos(k*M_PI/xpoints)+b;
}

inline double ChebyshevWeight (int k, double xmin, double xmax, int xpoints)
{
	if (xpoints==1) {return xmax-xmin;} // 0th order: can integrate constant at best, result = xmax-xmin
	double a = (xmax-xmin)*0.5;
	return a * M_PI/xpoints * sin((k+0.5)*M_PI/xpoints);
}

		// default: xmin=-1, xmax=+1
		inline double ChebyshevWeight (int k, int xpoints)
		{
			if (xpoints==1) {return 2.;} // 0th order: can integrate constant at best, result = 1-(-1)=2
			return M_PI/xpoints * sin((k+0.5)*M_PI/xpoints);
		}

inline double ChebyshevXpWeight (int k, double xmin, double xmax, int xpoints)
{
	double a = (xmax-xmin)*0.5;
	return a/xpoints;
}

		// default: xmin=-1, xmax=+1
		inline double ChebyshevXpWeight (int k, int xpoints)
		{
			return 1./xpoints;
		}

inline double ChebyshevPhase (int k, double xmin, double xmax, int xpoints)
{
	return (k+0.5)*M_PI/xpoints;
}
//----------------</Chebyshev-Gauss>----------------

//----------------<Chebyshev-Gauss-Lobatto>----------------
inline double ChebyshevLobattoAbscissa (int k, double xmin, double xmax, int xpoints)
{
	double a = (xmax-xmin)*0.5;
	double b = (xmax+xmin)*0.5;
	if (xpoints==1) {return b;}
	return -a*cos(k*M_PI/(xpoints-1))+b;
}

		// default: xmin=-1, xmax=+1
		inline double ChebyshevLobattoAbscissa (int k, int xpoints)
		{
			if (xpoints==1) {return 0;}
			return -cos(k*M_PI/(xpoints-1));
		}

inline double ChebyshevLobattoWeight (int k, double xmin, double xmax, int xpoints)
{
	if (xpoints==1) {return xmax-xmin;} // 0th order: can integrate constant at best, result = xmax-xmin
	else
	{
		double a = (xmax-xmin)*0.5;
		double res = a * M_PI/(xpoints-1) * sin(k*M_PI/(xpoints-1));
		if (k==0 || k==xpoints-1) {res *= 0.5;}
		return res;
	}
}

		// default: xmin=-1, xmax=+1
		inline double ChebyshevLobattoWeight (int k, int xpoints)
		{
			if (xpoints==1) {return 2.;} // 0th order: can integrate constant at best, result = 1-(-1)=2
			double res = M_PI/(xpoints-1) * sin(k*M_PI/(xpoints-1));
			if (k==0 || k==xpoints-1) {res *= 0.5;}
			return res;
		}
		
inline double ChebyshevLobattoXpWeight (int k, double xmin, double xmax, int xpoints)
{
	if (xpoints==1) {return xmax-xmin;}
	else
	{
		double a = (xmax-xmin)*0.5;
		if (k==0 || k==xpoints-1) {return a*0.5/(xpoints-1);}
		else {return a/(xpoints-1);}
	}
}

		// default: xmin=-1, xmax=+1
		inline double ChebyshevLobattoXpWeight (int k, int xpoints)
		{
			if (xpoints==1) {return 2.;}
			else
			{
				if (k==0 || k==xpoints-1) {return 0.5/(xpoints-1);}
				else                      {return 1./(xpoints-1);}
			}
		}

//****<differential weights>****
inline double ChebyshevLobattoDiffWeight (int k, double xmin, double xmax, int order)
{
	double res;
	if      (order<0)                          {res= 0;}
	else if (order==0)                         {res= ChebyshevLobattoWeight(k,xmin,xmax,1);}
	else if (order==1 && GSL_IS_ODD(k)==true)  {res= ChebyshevLobattoWeight(k,xmin,xmax,3)-ChebyshevLobattoWeight(k,xmin,xmax,1);}
	else if (order==1 && GSL_IS_ODD(k)==false) {res= ChebyshevLobattoWeight(k,xmin,xmax,3);}
	else
	{
		res= (GSL_IS_EVEN(k)==true) ? 
		ChebyshevLobattoWeight(k,xmin,xmax,pow(2,order)+1)-ChebyshevLobattoWeight(k/2,xmin,xmax,pow(2,order-1)+1) : 
		ChebyshevLobattoWeight(k,xmin,xmax,pow(2,order)+1);
	}
	return res;
}

inline double ChebyshevLobattoDiffWeight (int k, int order)
{
	if      (order<0)                          {return 0;}
	else if (order==0)                         {return ChebyshevLobattoWeight(k,1);}
	else if (order==1 && GSL_IS_ODD(k)==true)  {return ChebyshevLobattoWeight(k,3)-ChebyshevLobattoWeight(k,1);}
	else if (order==1 && GSL_IS_ODD(k)==false) {return ChebyshevLobattoWeight(k,3);}
	else
	{
		return (GSL_IS_EVEN(k)==true) ? 
		ChebyshevLobattoWeight(k,pow(2,order)+1)-ChebyshevLobattoWeight(k/2,pow(2,order-1)+1) : 
		ChebyshevLobattoWeight(k,pow(2,order)+1);
	}
}

inline double ChebyshevLobattoDiffXpWeight (int k, double xmin, double xmax, int order)
{
	if      (order<0)                          {return 0;}
	else if (order==0)                         {return ChebyshevLobattoXpWeight(k,xmin,xmax,1);}
	else if (order==1 && GSL_IS_ODD(k)==true)  {return ChebyshevLobattoXpWeight(k,xmin,xmax,3)-ChebyshevLobattoXpWeight(k,xmin,xmax,1);}
	else if (order==1 && GSL_IS_ODD(k)==false) {return ChebyshevLobattoXpWeight(k,xmin,xmax,3);}
	else
	{
		return (GSL_IS_EVEN(k)==true) ? 
		ChebyshevLobattoXpWeight(k,xmin,xmax,pow(2,order)+1)-ChebyshevLobattoXpWeight(k/2,xmin,xmax,pow(2,order-1)+1) : 
		ChebyshevLobattoXpWeight(k,xmin,xmax,pow(2,order)+1);
	}
}
//****</differential weights>****

inline double ChebyshevLobattoPhase (int k, double xmin, double xmax, int xpoints)
{
	if (xpoints==1) {return 0.5*M_PI;}
	else {return k*M_PI/(xpoints-1);}
}
//----------------</Chebyshev-Gauss-Lobatto>----------------

////----------------<Clenshaw-Curtis>---------------

//Eigen::VectorXd weightsCC;

//void calculate_weightsCC (int xpoints)
//{
//	int Nfft = (xpoints-1)/2+1;
//	weightsCC.resize(Nfft);
//	for (int k=0; k<Nfft; ++k)
//	{
//		weightsCC(k) = 2.*pow(xpoints-1.,-1)*pow(1.-4.*k*k,-1);
//	}
//	
//	#ifndef DONT_USE_OPENMP
//	int fftw_init_threads(void);
//	fftw_plan_with_nthreads(omp_get_max_threads());
//	#endif
//	fftw_plan plan = fftw_plan_r2r_1d(Nfft, weightsCC.data(),weightsCC.data(), FFTW_REDFT00,FFTW_ESTIMATE);
//	fftw_execute(plan);
//	fftw_destroy_plan(plan);
//	
//	weightsCC(0) = 1./(xpoints*(xpoints-2.));
//}

//inline double ClenshawCurtisWeight (int k, int xpoints)
//{
//	if (weightsCC.rows()!=xpoints) {calculate_weightsCC(xpoints);}
//	int Nfft = (xpoints-1)/2+1;
//	if (k>=Nfft) {return weightsCC(xpoints-k-1);}
//	else         {return weightsCC(k);}
//}

////----------------</Clenshaw-Curtis>---------------

//----------------<explicit Chebyshev polynomials>----------------
//double ChebyshevT (int n, double x)
//{
//	if      (n==0) {return 1.;}
//	else if (n==1) {return x;}
//	else if (n==2) {return 2.*pow(x,2)-1.;}
//	else if (n==3) {return 4.*pow(x,3)-3.*x;}
////	else if (n==4) {return 8.*pow(x,4)-8.*x*x+1.;}
////	else if (n==5) {return 16.*pow(x,5)-20.*pow(x,3)+5.*x;}
////	else if (n==6) {return 32.*pow(x,6)-48.*pow(x,4)+18.*pow(x,2)-1.;}
////	else if (n==7) {return 64.*pow(x,7)-112.*pow(x,5)+56.*pow(x,3)-7.*x;}
////	else if (n==8) {return 128.*pow(x,8)-256.*pow(x,6)+160.*pow(x,4)-32.*pow(x,2)+1.;}
////	else if (n==9) {return 256.*pow(x,9)-576.*pow(x,7)+432.*pow(x,5)-120.*pow(x,3)+9.*x;}
//	
//	double tnm1 = 4.*pow(x,3)-3.*x;
//	double tnm2 = 2.*pow(x,2)-1.;
//	double tn = tnm1;
//	
//	for (int l=4; l<=n; ++l)
//	{
//		tn = 2.*x*tnm1-tnm2;
//		tnm2 = tnm1;
//		tnm1 = tn;
//	}
//	return tn;
//}

//double ChebyshevTderiv (int n, double x)
//{
//	if      (n==0) {return 0.;}
//	else if (n==1) {return 1;}
//	else if (n==2) {return 4.*x-1.;}
//	else if (n==3) {return 12.*pow(x,2)-3.;}
//	else if (n==4) {return 32.*pow(x,3)-16.*x;}
//	
//	double tnm1 = 32.*pow(x,3)-16.*x;
//	double tnm2 = 12.*pow(x,2)-3.;
//	double tn = tnm1;
//	
//	for (int l=5; l<=n; ++l)
//	{
//		tn = 2.*x*tnm1-tnm2;
//		tnm2 = tnm1;
//		tnm1 = tn;
//	}
//	return tn;
//}
//----------------</explicit Chebyshev polynomials>----------------

#endif

#ifndef MY_GSL_INTEGRATION
#define MY_GSL_INTEGRATION

#ifndef GSL_WORKSPACE_SIZE
#define GSL_WORKSPACE_SIZE 1e3
#endif

#include <math.h>
#include <complex.h>
#include <gsl/gsl_integration.h>
using namespace std;

// Taken from: https://stackoverflow.com/questions/13289311/c-function-pointers-with-c11-lambdas/18413206#18413206
template<typename F> class gsl_function_pp : public gsl_function
{
public:
	
	gsl_function_pp (const F& func)
	:_func(func)
	{
		function = &gsl_function_pp::invoke;
		params = this;
	}
	
private:
	
	const F& _func;
	static double invoke (double x, void *params)
	{
		return static_cast<gsl_function_pp*>(params)->_func(x);
	}
};

struct integration_params
{
	double xl; // left x-value
	double xr; // right x-value
	double yl; // left y-value
	double yr; // right y-value
	double zl; // left z-value
	double zr; // right z-value
	
	double err_rel; // relative error
	double err_abs; // absolute error
	
	int GKrule; // Gauss-Kronrod rule (= 1 to 6)
	
	double (*function)(double,double); // pointer to function
	double (*f3d)(double,double,double); // pointer to function
};

// parameters needed for f1d, see below
struct param_y
{
	double (*function)(double,double);
	double y_value;
};

//function where y-value converted into parameter to integrate over x
double f1d (double x, void *params)
{
	param_y current_param = *(param_y *)params;
	return current_param.function(x, current_param.y_value);
}

// integrate over x with a fixed y converted into parameter (f1d)
double integrate_out_x (double y, void *params)
{
	gsl_integration_workspace * wx = gsl_integration_workspace_alloc(GSL_WORKSPACE_SIZE);
	gsl_function Fx;
	Fx.function = &f1d;
	
	integration_params param_readout;
	param_readout = *(integration_params*)params;
	param_y param_redirect;
	param_redirect.y_value = y;
	param_redirect.function = param_readout.function;
	Fx.params = &param_redirect;
	
	double res_x, err_x;
	gsl_integration_qag(&Fx, param_readout.xl, param_readout.xr, param_readout.err_abs, param_readout.err_rel, GSL_WORKSPACE_SIZE, param_readout.GKrule, wx, &res_x, &err_x);
	gsl_integration_workspace_free(wx);
	return res_x;
}

// integrate over y: integrate out x for each value
double integrate (double (*f_in)(double,double), integration_params param_in)
{
	gsl_integration_workspace * wy = gsl_integration_workspace_alloc(GSL_WORKSPACE_SIZE);
	gsl_function Fy;
	Fy.function = &integrate_out_x;
	
	param_in.function = f_in;
	if (param_in.GKrule==0) {param_in.GKrule=6;} //default value when no GK rule specified
	Fy.params = &param_in;
	
	double result, error;
	gsl_integration_qag (&Fy, param_in.yl, param_in.yr, param_in.err_abs, param_in.err_rel, GSL_WORKSPACE_SIZE, param_in.GKrule, wy, &result, &error);
	gsl_integration_workspace_free(wy);
	return result;
}

double integrate (double (*f_in)(double,double), double xl, double xr, double yl, double yr, double err_rel, double err_abs, int GKrule=6)
{
	integration_params param_transfer;
	param_transfer.xl=xl;
	param_transfer.xr=xr;
	param_transfer.yl=yl;
	param_transfer.yr=yr;
	param_transfer.err_rel=err_rel;
	param_transfer.err_abs=err_abs;
	param_transfer.GKrule=GKrule;
	return integrate(f_in,param_transfer);
}

// integrate: typing parameters into argument without GKrule and with identical range
double integrate (double (*f_in)(double,double), double l, double r, double err_rel, double err_abs)
{
	integration_params param_transfer;
	param_transfer.xl=l;
	param_transfer.xr=r;
	param_transfer.yl=l;
	param_transfer.yr=r;
	param_transfer.err_rel=err_rel;
	param_transfer.err_abs=err_abs;
	param_transfer.GKrule=6;
	return integrate(f_in,param_transfer);
}

// one-dimensional integration
double integrate (double (*f_in)(double,void*), integration_params param_in)
{
	gsl_integration_workspace * w = gsl_integration_workspace_alloc(GSL_WORKSPACE_SIZE);
	gsl_function F;
	F.function = f_in;
	
	double res_x, err_x;
	gsl_integration_qag(&F, param_in.xl, param_in.xr, param_in.err_abs, param_in.err_rel, GSL_WORKSPACE_SIZE, param_in.GKrule, w, &res_x, &err_x);
	gsl_integration_workspace_free(w);
	return res_x;
}

// one-dimensional integration : typing parameters into argument including GKrule
double integrate (double (*f_in)(double,void*), double xl, double xr, double yl, double yr, double err_rel, double err_abs, double GKrule)
{
	integration_params param_transfer;
	param_transfer.xl=xl;
	param_transfer.xr=xr;
	param_transfer.yl=yl;
	param_transfer.yr=yr;
	param_transfer.err_rel=err_rel;
	param_transfer.err_abs=err_abs;
	param_transfer.GKrule=GKrule;
	return integrate(f_in,param_transfer);
}

// one-dimensional integration : typing parameters into argument without GKrule
double integrate (double (*f_in)(double,void*), double xl, double xr, double yl, double yr, double err_rel, double err_abs)
{
	integration_params param_transfer;
	param_transfer.xl=xl;
	param_transfer.xr=xr;
	param_transfer.yl=yl;
	param_transfer.yr=yr;
	param_transfer.err_rel=err_rel;
	param_transfer.err_abs=err_abs;
	param_transfer.GKrule=6;
	return integrate(f_in,param_transfer);
}

// one-dimensional integration : typing parameters into argument without GKrule and with identical range
double integrate (double (*f_in)(double,void*), double l, double r, double err_rel, double err_abs)
{
	integration_params param_transfer;
	param_transfer.xl=l;
	param_transfer.xr=r;
	param_transfer.yl=l;
	param_transfer.yr=r;
	param_transfer.err_rel=err_rel;
	param_transfer.err_abs=err_abs;
	param_transfer.GKrule=6;
	return integrate(f_in,param_transfer);
}

//---------------<3dim>-------------------
struct param_yz
{
	double (*function)(double,double,double);
	double y_value;
	double z_value;
};

double f1d_yz (double x, void *params)
{
	param_yz current_param = *(param_yz*)params;
	return current_param.function(x, current_param.y_value, current_param.z_value);
}

double integrate_out_x (double y, double z, integration_params param_in)
{
	gsl_integration_workspace * wx = gsl_integration_workspace_alloc(GSL_WORKSPACE_SIZE);
	gsl_function Fx;
	Fx.function = &f1d_yz;
	
	param_yz param_redirect;
	param_redirect.y_value = y;
	param_redirect.z_value = z;
	param_redirect.function = param_in.f3d;
	Fx.params = &param_redirect;
	
	double res_x, err_x;
	gsl_integration_qag(&Fx, param_in.xl, param_in.xr, param_in.err_abs, param_in.err_rel, GSL_WORKSPACE_SIZE, param_in.GKrule, wx, &res_x, &err_x);
	gsl_integration_workspace_free(wx);
	return res_x;
}

struct param_z
{
	double (*function)(double,double,double);
	double (*function_xint)(double,double,integration_params);
	double z_value;
	integration_params params;
};

double f1d_z (double y, void *params)
{
	param_z current_param = *(param_z*)params;
	return current_param.function_xint(y, current_param.z_value, current_param.params);
}

double integrate_out_y (double z, void *params)
{
	gsl_integration_workspace * wy = gsl_integration_workspace_alloc(GSL_WORKSPACE_SIZE);
	gsl_function Fy;
	Fy.function = &f1d_z;

	integration_params param_readout = *(integration_params*)params;
	param_z param_redirect;
	param_redirect.z_value = z;
	param_redirect.function = param_readout.f3d;
	param_redirect.function_xint = integrate_out_x;
	param_redirect.params = param_readout;
	Fy.params = &param_redirect;
	
	double res_y, err_y;
	gsl_integration_qag(&Fy, param_readout.yl, param_readout.yr, param_readout.err_abs, param_readout.err_rel, GSL_WORKSPACE_SIZE, param_readout.GKrule, wy, &res_y, &err_y);
	gsl_integration_workspace_free(wy);
	return res_y;
}

double integrate (double (*f_in)(double,double,double), integration_params param_in)
{
	gsl_integration_workspace * wz = gsl_integration_workspace_alloc(GSL_WORKSPACE_SIZE);
	gsl_function Fz;
	Fz.function = &integrate_out_y;
	
	param_in.f3d = f_in;
	Fz.params = &param_in;
	
	double result, error;
	gsl_integration_qag (&Fz, param_in.zl, param_in.zr, param_in.err_abs, param_in.err_rel, GSL_WORKSPACE_SIZE, param_in.GKrule, wz, &result, &error);
	gsl_integration_workspace_free(wz);
	return result;
}

double integrate (double (*f_in)(double,double,double), double xl, double xr, double yl, double yr, double zl, double zr, double err_rel, double err_abs, int GKrule=6)
{
	integration_params param_transfer;
	param_transfer.xl=xl;
	param_transfer.xr=xr;
	param_transfer.yl=yl;
	param_transfer.yr=yr;
	param_transfer.zl=zl;
	param_transfer.zr=zr;
	param_transfer.err_rel=err_rel;
	param_transfer.err_abs=err_abs;
	param_transfer.GKrule=GKrule;
	return integrate(f_in,param_transfer);
}
//---------------</3dim>-------------------

// one-dimensional integration on infinite interval
double infigrate (double (*f_in)(double,void*), double err_rel, double err_abs)
{
	gsl_integration_workspace * w = gsl_integration_workspace_alloc(GSL_WORKSPACE_SIZE);
	gsl_function F;
	F.function = f_in;
	
	double res_x, err_x;
	gsl_integration_qagi(&F, err_abs, err_rel, GSL_WORKSPACE_SIZE, w, &res_x, &err_x);
	gsl_integration_workspace_free(w);
	return res_x;
}

// one-dimensional integration on infinite interval
double seminfigrate (double (*f_in)(double,void*), double xl, double err_rel, double err_abs)
{
	gsl_integration_workspace * w = gsl_integration_workspace_alloc(GSL_WORKSPACE_SIZE);
	gsl_function F;
	F.function = f_in;
	
	double res_x, err_x;
	gsl_integration_qagiu(&F, xl, err_abs, err_rel, GSL_WORKSPACE_SIZE, w, &res_x, &err_x);
	gsl_integration_workspace_free(w);
	return res_x;
}

// integrate real function f(t) times complex exp(i*omega*t)
double fouriergrate (double (*f_in)(double, void*), double l, double r, double err_rel, double err_abs, double omega)
{
	gsl_integration_workspace * w = gsl_integration_workspace_alloc(GSL_WORKSPACE_SIZE);
	gsl_integration_qawo_table * w_cos = gsl_integration_qawo_table_alloc (omega, r-l, GSL_INTEG_COSINE, GSL_WORKSPACE_SIZE);
	gsl_function F;
	F.function = f_in;
	
	double resRe, errRe;
	gsl_integration_qawo (&F, l, err_abs, err_rel, GSL_WORKSPACE_SIZE, w, w_cos, &resRe, &errRe);
	return resRe;
}

//***********************************COMPLEX**************************************************

class GSL_COMPLEX_wrapper
{
public:

	static double wrap_around_gsl_real (double x, void *params) {return function_real(x);}

	static double wrap_around_gsl_Re (double x, void *params) {return function(x).real();}
	static double wrap_around_gsl_Im (double x, void *params) {return function(x).imag();}
	
	static double wrap_around_gsl_Re2dim (double x, double y) {return function2dim(x,y).real();}
	static double wrap_around_gsl_Im2dim (double x, double y) {return function2dim(x,y).imag();}

	static void set_function_real (double (*function_input_real)(double)) {function_real = function_input_real;}
	static void set_function (complex<double> (*function_input)(double)) {function = function_input;}
	static void set_function2dim (complex<double> (*function2dim_input)(double,double)) {function2dim = function2dim_input;}

	static double          (*function_real)(double);
	static complex<double> (*function)(double);
	static complex<double> (*function2dim)(double,double);
};

double          (*GSL_COMPLEX_wrapper::function_real)(double) = NULL;
complex<double> (*GSL_COMPLEX_wrapper::function)(double) = NULL;
complex<double> (*GSL_COMPLEX_wrapper::function2dim)(double,double) = NULL;

// one-dimensional complex integration : typing parameters into argument without GKrule and with identical range
complex<double> integrate(complex<double> (*f_in)(double), double l, double r, double err_rel, double err_abs)
{
	integration_params param_transfer;
	param_transfer.xl=l;
	param_transfer.xr=r;
	param_transfer.yl=l;
	param_transfer.yr=r;
	param_transfer.err_rel=err_rel;
	param_transfer.err_abs=err_abs;
	param_transfer.GKrule=6;
	GSL_COMPLEX_wrapper::set_function(f_in);
	return complex<double>(
				integrate(GSL_COMPLEX_wrapper::wrap_around_gsl_Re,param_transfer), 
				integrate(GSL_COMPLEX_wrapper::wrap_around_gsl_Im,param_transfer));
}

// two-dimensional complex integration : typing parameters into argument without GKrule and with identical range
complex<double> integrate(complex<double> (*f_in)(double,double), double xl, double xr, double yl, double yr, double err_rel, double err_abs)
{
	integration_params param_transfer;
	param_transfer.xl=xl;
	param_transfer.xr=xr;
	param_transfer.yl=yl;
	param_transfer.yr=yr;
	param_transfer.err_rel=err_rel;
	param_transfer.err_abs=err_abs;
	param_transfer.GKrule=6;
	GSL_COMPLEX_wrapper::set_function2dim(f_in);
	return complex<double>(
				integrate(GSL_COMPLEX_wrapper::wrap_around_gsl_Re2dim, param_transfer), 
				integrate(GSL_COMPLEX_wrapper::wrap_around_gsl_Im2dim, param_transfer));
}

// integrate complex function Re(f(t))+i(Imf(t)) times complex exp(i*omega*t)
complex<double> fouriergrate(complex<double> (*f_in)(double), double l, double r, double err_rel, double err_abs, double omega)
{
	gsl_integration_workspace * w = gsl_integration_workspace_alloc(GSL_WORKSPACE_SIZE);
	gsl_integration_qawo_table * w_sin = gsl_integration_qawo_table_alloc (omega, r-l, GSL_INTEG_SINE, GSL_WORKSPACE_SIZE);
	gsl_integration_qawo_table * w_cos = gsl_integration_qawo_table_alloc (omega, r-l, GSL_INTEG_COSINE, GSL_WORKSPACE_SIZE);
	GSL_COMPLEX_wrapper::set_function(f_in);
	gsl_function FRe, FIm;
	FRe.function = GSL_COMPLEX_wrapper::wrap_around_gsl_Re;
	FIm.function = GSL_COMPLEX_wrapper::wrap_around_gsl_Im;
	double resReCos, resImSin, errReCos, errImSin, resReSin, resImCos, errReSin, errImCos;
	gsl_integration_qawo (&FRe, l, err_abs, err_rel, GSL_WORKSPACE_SIZE, w, w_cos, &resReCos, &errReCos);
	gsl_integration_qawo (&FIm, l, err_abs, err_rel, GSL_WORKSPACE_SIZE, w, w_sin, &resImSin, &errImSin);
	gsl_integration_qawo (&FRe, l, err_abs, err_rel, GSL_WORKSPACE_SIZE, w, w_sin, &resReSin, &errReSin);
	gsl_integration_qawo (&FIm, l, err_abs, err_rel, GSL_WORKSPACE_SIZE, w, w_cos, &resImCos, &errImCos);
	gsl_integration_qawo_table_free(w_sin);
	gsl_integration_qawo_table_free(w_cos);
	gsl_integration_workspace_free(w);
	return complex<double>(resReCos-resImSin,resReSin+resImCos);
}

complex<double> fouriergrate(gsl_function &FRe, gsl_function &FIm, double l, double r, double err_rel, double err_abs, double omega)
{
	gsl_integration_workspace * w = gsl_integration_workspace_alloc(GSL_WORKSPACE_SIZE);
	gsl_integration_qawo_table * w_sin = gsl_integration_qawo_table_alloc (omega, r-l, GSL_INTEG_SINE, GSL_WORKSPACE_SIZE);
	gsl_integration_qawo_table * w_cos = gsl_integration_qawo_table_alloc (omega, r-l, GSL_INTEG_COSINE, GSL_WORKSPACE_SIZE);
	double resReCos, resImSin, errReCos, errImSin, resReSin, resImCos, errReSin, errImCos;
	gsl_integration_qawo (&FRe, l, err_abs, err_rel, GSL_WORKSPACE_SIZE, w, w_cos, &resReCos, &errReCos);
	gsl_integration_qawo (&FIm, l, err_abs, err_rel, GSL_WORKSPACE_SIZE, w, w_sin, &resImSin, &errImSin);
	gsl_integration_qawo (&FRe, l, err_abs, err_rel, GSL_WORKSPACE_SIZE, w, w_sin, &resReSin, &errReSin);
	gsl_integration_qawo (&FIm, l, err_abs, err_rel, GSL_WORKSPACE_SIZE, w, w_cos, &resImCos, &errImCos);
//	cout << "errors=" << errReCos << ", " << errImCos << ", " << resReSin << ", " << resImSin << endl;
	gsl_integration_qawo_table_free(w_sin);
	gsl_integration_qawo_table_free(w_cos);
	gsl_integration_workspace_free(w);
	return complex<double>(resReCos-resImSin,resReSin+resImCos);
}

#endif

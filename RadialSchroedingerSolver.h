#ifndef RADIALSCHROEDINGERSOLVER
#define RADIALSCHROEDINGERSOLVER

#include <Eigen/Dense>

#include "OrthPolyEval.h"
#include "ChebyshevAbscissa.h"
#include "LanczosSolver.h"

struct less_than_energy
{
	inline bool operator() (const Eigenstate<Eigen::VectorXcd>& s1, const Eigenstate<Eigen::VectorXcd>& s2)
	{
		return (s1.energy < s2.energy);
	}
};

double intT (int n, double t)
{
	OrthPoly<CHEBYSHEV> ChebT;
	double res = 0;
	if (n==0)
	{
		return t+1.;
	}
	else if (n==1)
	{
		return 0.5*(t*t-1.);
	}
	else
	{
		return -((n+1)*t*ChebT.eval(n,t)-n*ChebT.eval(n+1,t)+pow(-1.,n))/(n*n-1.);
	}
	return res;
}

class RadialSchroedingerSolver
{
public:
	
	RadialSchroedingerSolver(){};
	
	RadialSchroedingerSolver (double rmin_input, double rmax_input, int Nmom_input)
	:rmin(rmin_input), rmax(rmax_input), Nmom(Nmom_input)
	{
		a = 0.5*(rmax-rmin);
		b = 0.5*(rmax+rmin);
		
		OrthPoly<CHEBYSHEV> ChebT;
		
		D.resize(Nmom,Nmom); D.setZero();
		for (int i=0; i<Nmom; ++i)
		for (int j=0; j<Nmom; ++j)
		{
			double xi = ChebyshevLobattoAbscissa(i,Nmom);
			double xj = ChebyshevLobattoAbscissa(j,Nmom);
			for (int k=0; k<Nmom; ++k)
			{
				D(i,j) += 2.*ChebyshevLobattoXpWeight(k,Nmom) * ChebT.eval(k,xj) * intT(k,xi);
			}
		}
		D.col(0) *= 0.5;
		D.col(D.cols()-1) *= 0.5;
		if (Nmom<=10) lout << D << endl;
		d = D.row(D.rows()-1);
		
		E = D*D;
		e = E.row(E.rows()-1);
		
		x.resize(Nmom);
		u.resize(Nmom);
		for (int i=0; i<Nmom; ++i)
		{
			x(i) = ChebyshevLobattoAbscissa(i,Nmom);
			u(i) = 0.5*(x(i)+1.);
		}
	};
	
	inline Eigen::ArrayXd r() const
	{
		return a*x.array()+b;
	}
	
	inline double integrate (const VectorXd &vec) const
	{
		return d.dot(vec)*a;
	}
	
	vector<Eigenstate<Eigen::VectorXcd>> bound_states (double (*V)(double), double Escale=1.)
	{
		Eigen::VectorXd Vvec(Nmom);
		for (int i=0; i<Nmom; ++i)
		{
			double r = a*x(i)+b;
			Vvec(i) = a*a* V(r);
		}
		
		Eigen::MatrixXd Dhat(Nmom,Nmom); Dhat.setZero();
		for (int i=0; i<Nmom; ++i)
		for (int j=0; j<Nmom; ++j)
		{
			Dhat(i,j) = D(i,j) * Vvec(j);
		}
		
		Eigen::VectorXd s = (D*Dhat).row(D.rows()-1);
		
		Eigen::MatrixXd Ehat(Nmom,Nmom); Ehat.setZero();
		for (int i=0; i<Nmom; ++i)
		for (int j=0; j<Nmom; ++j)
		{
			Ehat(i,j) = E(i,j) * Vvec(j);
		}
		
		Eigen::MatrixXd Amat(Nmom,Nmom); Amat.setZero();
		for (int i=0; i<Nmom; ++i)
		for (int j=0; j<Nmom; ++j)
		{
			Amat(i,j) -= u(i)*s(j);
			if (i==j)
			{
				Amat(i,j) -= 1.;
			}
		}
		Amat += Ehat;
		
		Eigen::MatrixXd Bmat = E;
		for (int i=0; i<Nmom; ++i)
		for (int j=0; j<Nmom; ++j)
		{
			Bmat(i,j) -= u(i)*e(j);
		}
		
		Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> General(Amat,Bmat);
		
		//	EigenSolver<MatrixXd> Eugen(A.inverse()*B);
		//	cout << "eigenvalues=" << endl << 4./Eugen.eigenvalues().array() << endl;
		//	cout << endl;
		
		vector<Eigenstate<Eigen::VectorXcd>> res;
		
		for (int i=0; i<General.eigenvalues().rows(); ++i)
		{
			if (General.eigenvalues()(i).real() < 0. and not std::isinf(abs(General.eigenvalues()(i).real())))
			{
				Eigenstate<Eigen::VectorXcd> bound_solution;
				bound_solution.energy = Escale*General.eigenvalues()(i).real()/pow(a,2);
				bound_solution.state = General.eigenvectors().col(i);
				res.push_back(bound_solution);
			}
		}
		sort(res.data(), res.data()+res.size(), less_than_energy());
		
		return res;
	}
	
private:
	
	Eigen::VectorXd d, e, x, u;
	Eigen::MatrixXd D, E;
	
	double rmin, rmax, a, b;
	int Nmom;
};

#endif

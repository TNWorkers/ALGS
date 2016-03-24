#ifndef ORTHPOLYGREEN1DIM
#define ORTHPOLYGREEN1DIM

#include "OrthPolyBase.h"
#include "IntervalIterator.h"
#include "MemCalc.h"
#include "LinearPrediction.h"
#include "ChebyshevAbscissa.h"

template<typename Hamiltonian, typename VectorType, ORTHPOLY P=CHEBYSHEV>
class OrthPolyGreen : public OrthPolyBase<Hamiltonian,VectorType>
{
public:
	
	OrthPolyGreen (const Hamiltonian &H, double padding_input=0.005);
	OrthPolyGreen (double Emin_input, double Emax_input, double padding_input=0.005);
	
	string info() const;
	
	void calc_ImAA (Hamiltonian &H, const VectorType &OxV_init, int M_input, bool USE_IDENTITIES=true);
//	template<typename StateIterator> void calc_ImAB (Hamiltonian &H, StateIterator &AxV, const VectorType &BxV, int M_input);
	void calc_ImAB (Hamiltonian &H, const vector<VectorType> &AxV, const VectorType &BxV, int M_input);
	
	// integrals, evaluations
	double ImAAintegral (double (*f)(double), int Msave, double Eoffset=0., bool REVERSE=false, KERNEL_CHOICE KERNEL=JACKSON);
	double ImAAarea (KERNEL_CHOICE KERNEL=JACKSON);
	double evaluate_ImAA (double E, int Msave=-1, double Eoffset=0., bool REVERSE=false, KERNEL_CHOICE KERNEL=JACKSON);
	double evaluate_ImAA_scaled (double x, int Msave_input, bool REVERSE=false, KERNEL_CHOICE KERNEL=JACKSON);
//	double evaluate_deriv (double E, int Msave_input, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL=JACKSON);
	double evaluate_ImAB (int i, double E, int Msave_input=-1, double Eoffset=0., bool REVERSE=false, KERNEL_CHOICE KERNEL=JACKSON);
	double evaluate_ImAA_Chebyshev (double E, int Msave=-1, double Eoffset=0., bool REVERSE=false, KERNEL_CHOICE KERNEL=JACKSON);
	double ImAAselfconv (double y, int Msave=-1, double Eoffset=0., bool REVERSE=false, KERNEL_CHOICE KERNEL=JACKSON);
	
	// saving, returning, injecting
	void add_savepoint (int Msave, string momfile, string datfile, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL=JACKSON);
	void clear_savepoints() {savepoints.clear();};
	
	void save_ImAA (int Msave, string dumpfile, double Eoffset=0., bool REVERSE=false, KERNEL_CHOICE KERNEL=JACKSON);
//	void save_ImAAderiv (int Msave, string datfile, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL=JACKSON);
	void save_ImAAmoments (string momfile, int Msave_input=-1);
	void inject_ImAAmoments (const VectorXd ImAAmoments_input);
	MatrixXd get_ImAA (int Msave, double Eoffset=0., bool REVERSE=false, KERNEL_CHOICE KERNEL=JACKSON);
	VectorXd get_ImAAmoments() {return ImAAmoments;}
	
	void save_ImAB (int Msave, string datfile, double Eoffset=0., bool REVERSE=false, KERNEL_CHOICE KERNEL=JACKSON);
	void save_ImABmoments (string momfile, int Msave_input=-1);
	
	void predict_ImAA (int M_new);
	
	inline int get_M() {return M;}
	
	void calc_ImAA_Legendre (Hamiltonian &H, const VectorType &AxV, int M_input, bool USE_IDENTITIES);
	double evaluate_Legendre (double x, int Msave=-1, double Eoffset=0., bool REVERSE=false, KERNEL_CHOICE KERNEL=JACKSON);
	void save_ImAA_Legendre (int Msave, string datfile, double Eoffset=0., bool REVERSE=false, KERNEL_CHOICE KERNEL=JACKSON);
	
private:
	
	list<tuple<int,string,string,double,bool,KERNEL_CHOICE> > savepoints;
	
	int M;
	VectorXd ImAAmoments;
	bool GOT_MOMENTS;
	
	vector<VectorXd> ImABmoments;
	int Asize;
};

template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
OrthPolyGreen<Hamiltonian,VectorType,P>::
OrthPolyGreen (const Hamiltonian &H, double padding_input)
:OrthPolyBase<Hamiltonian,VectorType>(H,padding_input), GOT_MOMENTS(false)
{}

template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
OrthPolyGreen<Hamiltonian,VectorType,P>::
OrthPolyGreen (double Emin_input, double Emax_input, double padding_input)
:OrthPolyBase<Hamiltonian,VectorType>(Emin_input,Emax_input,padding_input), GOT_MOMENTS(false)
{}

template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
string OrthPolyGreen<Hamiltonian,VectorType,P>::
info() const
{
//	stringstream ss;
//	ss << "KPS_ODOS:"
//	   << " Emin=" << this->Emin 
//	   << ", Emax=" << this->Emax
//	   << ", ½width a=" << this->a
//	   << ", centre b=" << this->b;
//	if (GOT_MOMENTS == true)
//	{
//		ss << ", mvms=" << this->N_mvm
//		   << ", moments=" << M;
//	}
	stringstream label;
	label << "KernelPolynomialSolver<" << P << ">";
	stringstream ss;
	ss << this->baseinfo(label.str());
	return ss.str();
}

template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
void OrthPolyGreen<Hamiltonian,VectorType,P>::
add_savepoint (int Msave, string momfile, string datfile, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL)
{
	tuple<int,string,string,double,bool,KERNEL_CHOICE> info = make_tuple(Msave,momfile,datfile,Eoffset,REVERSE,KERNEL);
	savepoints.push_back(info);
}

// -1/π Im≪A;A≫(ω)
template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
void OrthPolyGreen<Hamiltonian,VectorType,P>::
calc_ImAA (Hamiltonian &H, const VectorType &AxV, int M_input, bool USE_IDENTITIES)
{
	M = M_input;
	assert(M >= 2 and "Need at least 2 Chebyshev moments!");
	if (USE_IDENTITIES == true)
	{
		assert(M%2==0 and "Need an even amound of Chebyshev moments when exploiting the T_2n identities!");
	}
	
	ImAAmoments.resize(M);
	ImAAmoments.setZero();
	
	lout << "****** -1/π Im≪A†;A≫(ω) iteration: " << "0+1" << " ******" << endl;
	
	VectorType V0 = AxV;
	VectorType V1;
	H.scale(this->alpha,this->beta); // H = α·H+β
	HxV(H,V0,V1); ++this->N_mvm; // V1 = H·V0;
	
	ImAAmoments(0) = dot(V0,V0);
	ImAAmoments(1) = dot(V0,V1);
	
	H.scale(OrthPoly<P>::C(1)); // H = C_n·(α·H+β)
	
	VectorType Vtmp;
	int range = (USE_IDENTITIES==true)? M/2 : M-1;
	for (int n=1; n<range; ++n)
	{
		lout << "****** -1/π Im≪A†;A≫(ω) iteration: " << 1+n << " / " << range << " ******" << endl;
		
		H.scale(OrthPoly<P>::C(n+1)/OrthPoly<P>::C(n)); // H = C_{n+1}·(α·H+β)
		polyIter(H,V1,OrthPoly<P>::B(n+1),V0,Vtmp); ++this->N_mvm; // Vtmp = C_{n+1}(α·H+β)·V1 - B_{n+1}·V0
		V0 = V1; // V0 = V_{n-1}
		V1 = Vtmp; // V1 = V_{n}
		
		if (USE_IDENTITIES == true)
		{
			ImAAmoments(2*n)   = 2.*dot(V0,V0)-ImAAmoments(0);
			ImAAmoments(2*n+1) = 2.*dot(V1,V0)-ImAAmoments(1);
		}
		else
		{
			ImAAmoments(n+1) = dot(AxV,V1);
		}
		
		//----<saving>----
		auto save = [this](int Mcurr)
		{
			for (auto info=savepoints.begin(); info!=savepoints.end(); ++info)
			{
				if (get<0>(*info) == Mcurr)
				{
					save_ImAAmoments(get<1>(*info),Mcurr);
					save_ImAA(Mcurr, get<2>(*info), get<3>(*info), get<4>(*info), get<5>(*info));
				}
			}
		};
		
		if (USE_IDENTITIES == true)
		{
			save(2*n+1);
			save(2*n+2);
		}
		else
		{
			save(n+2);
		}
		//----</saving>----
	}
	
	H.scale(this->a/OrthPoly<P>::C(range),this->b);
	
	GOT_MOMENTS = true;
}

// -1/π Im≪A;B≫(ω)
template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
//template<typename StateIterator>
void OrthPolyGreen<Hamiltonian,VectorType,P>::
calc_ImAB (Hamiltonian &H, const vector<VectorType> &AxV, const VectorType &BxV, int M_input)
//calc_ImAB (Hamiltonian &H, StateIterator &AxV, const VectorType &BxV, int M_input)
{
	M = M_input;
	Asize = AxV.size();
	assert(M >= 2 and "Need at least 2 Chebyshev moments!");
	
	ImABmoments.resize(Asize);
	for (size_t i=0; i<Asize; ++i)
	{
		ImABmoments[i].resize(M);
		ImABmoments[i].setZero();
	}
	
	lout << "****** -1/π Im≪A;B≫(ω) iteration: " << "0+1" << " ******" << endl;
	
	VectorType V0 = BxV;
	VectorType V1;
	H.scale(this->alpha,this->beta); // H = α·H+β
	HxV(H,V0,V1); ++this->N_mvm; // V1 = H·V0;
	
	for (size_t i=0; i<Asize; ++i)
	{
		ImABmoments[i](0) = dot(AxV[i],V0);
		ImABmoments[i](1) = dot(AxV[i],V1);
	}
//	for (AxV=AxV.begin(); AxV!=AxV.end(); ++AxV)
//	{
//		ImABmoments[AxV.index()](0) = dot(*AxV,V0);
//		ImABmoments[AxV.index()](1) = dot(*AxV,V1);
//	}
	
	H.scale(OrthPoly<P>::C(1)); // H = A_n·(α·H+β)
	
	VectorType Vtmp;
	for (int n=1; n<M-1; ++n)
	{
		lout << "****** -1/π Im≪A;B≫(ω) iteration: " << 1+n << " / " << M-1 << " ******" << endl;
		
		H.scale(OrthPoly<P>::C(n+1)/OrthPoly<P>::C(n)); // H = A_{n+1}·(α·H+β)
		polyIter(H,V1,OrthPoly<P>::B(n+1),V0,Vtmp); ++this->N_mvm; // Vtmp = A_{n+1}(α·H+β)·V1 - B_{n+1}·V0
		V0 = V1; // V0 = V_{n-1}
		V1 = Vtmp; // V1 = V_{n}
		
		for (size_t i=0; i<Asize; ++i)
		{
			ImABmoments[i](n+1) = dot(AxV[i],V1);
		}
//		for (AxV=AxV.begin(); AxV!=AxV.end(); ++AxV)
//		{
//			ImABmoments[AxV.index()](n+1) = dot(*AxV,V1);
//		}
		
		//----<save>----
		for (auto info=savepoints.begin(); info!=savepoints.end(); ++info)
		{
			if (get<0>(*info) == n+2)
			{
				save_ImABmoments(get<1>(*info),n+2);
				save_ImAB(n+2, get<2>(*info), get<3>(*info), get<4>(*info), get<5>(*info));
			}
		}
		//----</save>----
	}
	H.scale(this->a/OrthPoly<P>::C(M),this->b);
	
	GOT_MOMENTS = true;
}

template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
void OrthPolyGreen<Hamiltonian,VectorType,P>::
save_ImAA (int Msave, string datfile, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL)
{
	assert(Msave <= M);
	double a=this->a; double b=this->b; double padding=this->padding;
	double Emin=this->Emin; double Emax=this->Emax;
	
	int Epoints = max(2*Msave,4000); // O(10^3) Epoints for smooth plots
	
//	IntervalIterator Eit(Emin-Eoffset,Emax-Eoffset,Epoints,ChebyshevAbscissa);
//	for (Eit=Eit.begin(); Eit<Eit.end(); ++Eit)
//	{
//		double E_scaled = (Eit.value()+Eoffset-b)/a;
//		double res = ImAAgamma(Eit.index())/(M_PI*sqrt(1.-E_scaled*E_scaled)*abs(a));
//		Eit << res;
//	}
//	Eit.save(datfile);
	
//	double DeltaE = Emax-Emin;
//	MatrixXd Esave(Epoints,2);
//	#pragma omp parallel for
//	for (int k=0; k<Epoints; ++k)
//	{
////		double Eval = ChebyshevAbscissa(k, Emin-0.025*DeltaE-Eoffset, Emax+0.025*DeltaE-Eoffset, Epoints);
//		double Eval = ChebyshevAbscissa(k, Emin-Eoffset, Emax-Eoffset, Epoints);
//		double specval = evaluate(Eval,Msave,Eoffset,REVERSE,KERNEL);
//		if (std::isnan(specval)) {specval = 0.;}
//		Esave(k,0) = Eval;
//		Esave(k,1) = specval;
//	}
//	ofstream datfiler(datfile);
//	datfiler << Esave << endl;
//	datfiler.close();
	
//	IntervalIterator Eit(-1.,+1.,Epoints,ChebyshevAbscissa);
//	ofstream datfiler(datfile);
//	for (Eit=Eit.begin(); Eit<Eit.end(); ++Eit)
//	{
//		if (abs(*Eit) < 1.-this->padding)
//		{
//			datfiler << *Eit << "\t" << ImAAgamma(Eit.index())/(M_PI*sqrt(1.-pow(*Eit,2))*abs(a)) << endl;
//		}
//	}
//	datfiler.close();
	
	ofstream datfiler(datfile);
	
	if (P == CHEBYSHEV)
	{
		IntervalIterator Eit(-1.,+1.,Epoints,ChebyshevAbscissa);
		VectorXd ImAAgamma = this->fct(ImAAmoments.head(Msave),Epoints,REVERSE,KERNEL);
		for (Eit=Eit.begin(); Eit<Eit.end(); ++Eit)
		{
			if (abs(*Eit) < 1.-this->padding)
			{
				datfiler << a*(*Eit)+b-Eoffset << "\t" << ImAAgamma(Eit.index())*OrthPoly<P>::w(*Eit)/abs(a) << endl;
			}
		}
	}
	else
	{
		IntervalIterator x(-1.,+1.,Epoints);
		for (x=x.begin(); x<x.end(); ++x)
		{
			if (abs(*x) < 1.-this->padding)
			{
				datfiler << a*(*x)+b-Eoffset << "\t" << evaluate_ImAA_scaled(*x,Msave,REVERSE,KERNEL) << endl;
			}
		}
	}
	datfiler.close();
	
	lout << datfile << " done!" << endl;
}

template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
void OrthPolyGreen<Hamiltonian,VectorType,P>::
save_ImAB (int Msave, string datfile, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL)
{
	assert(Msave <= M);
	double a=this->a; double b=this->b; double Emin=this->Emin; double Emax=this->Emax;
	
	int Epoints = max(2*Msave,4000); // O(10^3) Epoints for smooth plots
//	IntervalIterator Eit(Emin-Eoffset,Emax-Eoffset,Epoints,ChebyshevAbscissa);
//	
//	MatrixXd ImABspectrum(Epoints,Asize+1);
//	
//	ImABspectrum.col(0) = Eit.get_values();
//	
//	for (size_t i=0; i<Asize; ++i)
//	{
//		VectorXd ImABgamma = this->fct(ImABmoments[i].head(Msave),Epoints,REVERSE,KERNEL);
//		
//		for (Eit=Eit.begin(); Eit<Eit.end(); ++Eit)
//		{
//			double E_scaled = (Eit.value()+Eoffset-b)/a;
//			ImABspectrum(Eit.index(),i+1) = ImABgamma(Eit.index())/(M_PI*sqrt(1.-E_scaled*E_scaled)*abs(a));
//		}
//	}
	
//	IntervalIterator Eit(-1.,+1.,Epoints,ChebyshevAbscissa);
//	MatrixXd ImABspectrum;
//	
//	for (Eit=Eit.begin(); Eit<Eit.end(); ++Eit)
//	{
//		if (abs(*Eit) < 1.-this->padding)
//		{
//			ImABspectrum.conservativeResize(ImABspectrum.rows()+1,Asize+1);
//			ImABspectrum(ImABspectrum.rows()-1,0) = a*(*Eit)+b-Eoffset;
//		}
//	}
//	
//	for (size_t i=0; i<Asize; ++i)
//	{
//		VectorXd ImABgamma = this->fct(ImABmoments[i].head(Msave), Epoints, REVERSE, KERNEL);
//		
//		int iE = 0;
//		for (Eit=Eit.begin(); Eit<Eit.end(); ++Eit)
//		{
//			if (abs(*Eit) < 1.-this->padding)
//			{
//				ImABspectrum(iE,i+1) = ImABgamma(Eit.index())/(M_PI*sqrt(1.-pow(*Eit,2))*abs(a));
//				++iE;
//			}
//		}
//	}
	
	
	MatrixXd ImABspectrum(Epoints,Asize+1);
	
	IntervalIterator Eit_(Emin-Eoffset,Emax-Eoffset,Epoints);
	for (Eit_=Eit_.begin(); Eit_<Eit_.end(); ++Eit_)
	{
		ImABspectrum(Eit_.index(),0) = *Eit_;
	}
	
	#pragma omp parallel for
	for (size_t i=0; i<Asize; ++i)
	{
		IntervalIterator Eit(Emin-Eoffset,Emax-Eoffset,Epoints);
		for (Eit=Eit.begin(); Eit<Eit.end(); ++Eit)
		{
			ImABspectrum(Eit.index(),i+1) = evaluate_ImAB(i, *Eit, Msave, Eoffset, REVERSE, KERNEL);
		}
	}
	
	ofstream datfiler(datfile);
	datfiler << ImABspectrum << endl;
	datfiler.close();
	
	lout << datfile << " done!" << endl;
}

//template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
//void OrthPolyGreen<Hamiltonian,VectorType,P>::
//save_ImAAderiv (int Msave, string datfile, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL)
//{
//	assert(Msave <= M);
//	double a=this->a; double b=this->b; double padding=this->padding;
//	double Emin=this->Emin; double Emax=this->Emax;
//	int Epoints = max(2*Msave,4000); // O(10^3) Epoints for smooth plots
//	
//	IntervalIterator Eit(Emin-Eoffset,Emax-Eoffset,Epoints);
//	ofstream datfiler(datfile);
//	for (Eit=Eit.begin(); Eit<Eit.end(); ++Eit)
//	{
//		datfiler << *Eit << "\t" << evaluate_deriv(*Eit,Msave,Eoffset,REVERSE,KERNEL) << endl;
//	}
//	datfiler.close();
//	
////	IntervalIterator Eit(-1.,+1.,Epoints,ChebyshevAbscissa);
////	ofstream datfiler(datfile);
////	for (Eit=Eit.begin(); Eit<Eit.end(); ++Eit)
////	{
////		if (abs(*Eit) < 1.-this->padding)
////		{
////			datfiler << a*(*Eit)+b-Eoffset << "\t" << evaluate_deriv(a*(*Eit)+b-Eoffset,Msave,Eoffset,REVERSE,KERNEL) << endl;
////		}
////	}
////	datfiler.close();
//	
//	lout << datfile << " done!" << endl;
//}

template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
void OrthPolyGreen<Hamiltonian,VectorType,P>::
save_ImAAmoments (string momfile, int Msave_input)
{
	ofstream fout(momfile);
	int Msave = (Msave_input==-1)? ImAAmoments.rows() : Msave_input;
	
	for (size_t n=0; n<Msave; ++n)
	{
		fout << setprecision(13) << ImAAmoments(n) << endl;
	}
	
	fout.close();
	lout << momfile << " done!" << endl;
}

template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
void OrthPolyGreen<Hamiltonian,VectorType,P>::
save_ImABmoments (string momfile, int Msave_input)
{
	ofstream fout(momfile);
	int Msave = (Msave_input==-1)? ImAAmoments.rows() : Msave_input;
	
	for (size_t n=0; n<Msave; ++n)
	{
		for (size_t i=0; i<Asize; ++i)
		{
			fout << setprecision(13) << ImABmoments[i](n) << "\t";
		}
		fout << endl;
	}
	
	fout.close();
	lout << momfile << " done!" << endl;
}

template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
inline double OrthPolyGreen<Hamiltonian,VectorType,P>::
ImAAarea (KERNEL_CHOICE KERNEL)
{
	assert(GOT_MOMENTS == true and P==CHEBYSHEV);
	VectorXd ImAAgamma = this->fct(ImAAmoments,ImAAmoments.rows(),false,KERNEL);
	return ImAAgamma.sum()/ImAAmoments.rows();
}

// using normal recurrence
template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
double OrthPolyGreen<Hamiltonian,VectorType,P>::
evaluate_ImAA (double E, int Msave_input, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL)
{
	if (E+Eoffset<=this->Emin or E+Eoffset>=this->Emax) {return 0.;}
	else
	{
		int Msave = (Msave_input==-1)? ImAAmoments.rows() : Msave_input;
		double a=this->a; double b=this->b;
		
		double E_scaled = (E+Eoffset-b)/a;
		
		double res = OrthPoly<P>::orthfac(0) * ImAAmoments(0) * this->kernel(0,Msave,KERNEL) * OrthPoly<P>::eval(0,E_scaled);
		for (int n=1; n<Msave; ++n)
		{
			double phase = (REVERSE==true)? pow(-1.,n) : 1.;
			res += OrthPoly<P>::orthfac(n) * ImAAmoments(n) * this->kernel(n,Msave,KERNEL) * phase * OrthPoly<P>::eval(n,E_scaled);
		}
		
		return res*OrthPoly<P>::w(E_scaled)/abs(a);
	}
}

//template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
//double OrthPolyGreen<Hamiltonian,VectorType,P>::
//evaluate_ReAA (double E, int Msave_input, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL)
//{
//	if (E+Eoffset<=this->Emin or E+Eoffset>=this->Emax) {return 0.;}
//	else
//	{
//		int Msave = (Msave_input==-1)? ImAAmoments.rows() : Msave_input;
//		double a=this->a; double b=this->b;
//		
//		double E_scaled = (E+Eoffset-b)/a;
//		cout << "E_scaled = " << E_scaled << endl;
//		
//		double res = OrthPoly<P>::orthfac(0) * ImAAmoments(0) * this->kernel(0,Msave,KERNEL) * OrthPoly<P>::eval(0,E_scaled);
//		for (int n=1; n<Msave; ++n)
//		{
//			double phase = (REVERSE==true)? pow(-1.,n) : 1.;
//			res += OrthPoly<P>::orthfac(n) * ImAAmoments(n) * this->kernel(n,Msave,KERNEL) * phase * OrthPoly<P>::eval(n,E_scaled);
//		}
//		
//		return res*OrthPoly<P>::w(E_scaled)/abs(a);
//	}
//}

// using normal recurrence
template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
double OrthPolyGreen<Hamiltonian,VectorType,P>::
evaluate_ImAA_scaled (double x, int Msave_input, bool REVERSE, KERNEL_CHOICE KERNEL)
{
	if (x<-1. or x>1.) {return 0.;}
	else
	{
		int Msave = (Msave_input==-1)? ImAAmoments.rows() : Msave_input;
		double a=this->a; double b=this->b;
		
		double res = OrthPoly<P>::orthfac(0) * ImAAmoments(0) * this->kernel(0,Msave,KERNEL) * OrthPoly<P>::eval(0,x);
		for (int n=1; n<Msave; ++n)
		{
			double phase = (REVERSE==true)? pow(-1.,n) : 1.;
			res += OrthPoly<P>::orthfac(n) * ImAAmoments(n) * this->kernel(n,Msave,KERNEL) * phase * OrthPoly<P>::eval(n,x);
		}
		return res*OrthPoly<P>::w(x)/abs(a);
	}
}

// using Clenshaw recurrence
template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
double OrthPolyGreen<Hamiltonian,VectorType,P>::
evaluate_ImAA_Chebyshev (double E, int Msave_input, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL)
{
	assert(P == CHEBYSHEV);
	int Msave = (Msave_input==-1)? ImAAmoments.rows() : Msave_input;
	double a=this->a; double b=this->b; double Emin=this->Emin; double Emax=this->Emax;
	
	double E_scaled = (E+Eoffset-b)/a;
	VectorXd work = ImAAmoments;
	
	for (int n=0; n<Msave; ++n)
	{
		double phase = (REVERSE==true)? pow(-1.,n) : 1.;
		work(n) *= this->kernel(n,Msave,KERNEL) * phase;
	}
	
//	return ChebyshevExpansion(E_scaled,work)*(M_PI*sqrt(1.-E_scaled*E_scaled)*abs(a));
	return ChebyshevExpansion(E_scaled,work)*OrthPoly<P>::w(E_scaled)/abs(a);
}

//// using Chebyshev recurrence
//template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
//double OrthPolyGreen<Hamiltonian,VectorType,P>::
//evaluate_deriv (double E, int Msave_input, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL)
//{
//	if (E+Eoffset<=this->Emin or E+Eoffset>=this->Emax) {return 0.;}
//	else
//	{
//		int Msave = (Msave_input==-1)? ImAAmoments.rows() : Msave_input;
//		double a=this->a; double b=this->b;
//		
//		double E_scaled = (E+Eoffset-b)/a;
//		
//		double res = 0.;
//		for (int n=1; n<Msave; ++n)
//		{
//			double phase = (REVERSE==true)? pow(-1.,n) : 1.;
//			res += 2.*ImAAmoments(n) * this->kernel(n,Msave,KERNEL) * phase * ChebyshevTderiv(n,E_scaled);
//		}
//		
//		return res/(M_PI*sqrt(1.-E_scaled*E_scaled)*abs(a));
//	}
//}

// using Chebyshev recurrence
template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
double OrthPolyGreen<Hamiltonian,VectorType,P>::
evaluate_ImAB (int i, double E, int Msave_input, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL)
{
	int Msave = (Msave_input==-1)? ImABmoments[i].rows() : Msave_input;
	double a=this->a; double b=this->b;
	
	double E_scaled = (E+Eoffset-b)/a;

	double res = OrthPoly<P>::orthfac(0) * ImABmoments[i](0) * this->kernel(0,Msave,KERNEL) * OrthPoly<P>::eval(0,E_scaled);
	for (int n=1; n<Msave; ++n)
	{
		double phase = (REVERSE==true)? pow(-1.,n) : 1.;
//		res += 2.*ImABmoments[i](n) * this->kernel(n,Msave,KERNEL) * phase * ChebyshevT(n,E_scaled);
		res += OrthPoly<P>::orthfac(n) * ImABmoments[i](n) * this->kernel(n,Msave,KERNEL) * phase * OrthPoly<P>::eval(n,E_scaled);
	}
	
	return res*OrthPoly<P>::w(E_scaled)/abs(a);
}

template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
double OrthPolyGreen<Hamiltonian,VectorType,P>::
ImAAintegral (double (*f)(double), int Msave, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL)
{
	assert(GOT_MOMENTS == true and Msave <= M and P==CHEBYSHEV);
	double Emin=this->Emin; double Emax=this->Emax;
	
	VectorXd ImAAgamma = this->fct(ImAAmoments,Msave,REVERSE,KERNEL);
	IntervalIterator Eit(Emin-Eoffset,Emax-Eoffset,Msave,ChebyshevAbscissa);
	
	return (ImAAgamma.cwiseProduct(Eit.get_values().unaryExpr(std::ptr_fun(f)))).sum();
}

// \int A(x)*A(y-x)dx
template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
double OrthPolyGreen<Hamiltonian,VectorType,P>::
ImAAselfconv (double y, int Msave, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL)
{
	assert(GOT_MOMENTS == true and Msave <= M and P==CHEBYSHEV);
	double Emin=this->Emin; double Emax=this->Emax;
	
	int Epoints = max(2*Msave,4096);
	VectorXd ImAAgamma = this->fct(ImAAmoments,Epoints,REVERSE,KERNEL);
	IntervalIterator Eit(Emin-Eoffset,Emax-Eoffset,Epoints,ChebyshevAbscissa);
	
	double res = 0;
	#pragma omp parallel for reduction (+:res)
	for (int i=0; i<Epoints; ++i)
	{
		if (y-Eit(i) > Emin-Eoffset and y-Eit(i) < Emax-Eoffset)
		{
			res += ImAAgamma(i) * evaluate_ImAA(y-Eit(i),Msave,Eoffset,REVERSE,KERNEL);
		}
	}
	return res/Epoints;
}

template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
MatrixXd OrthPolyGreen<Hamiltonian,VectorType,P>::
get_ImAA (int Msave, double Eoffset, bool REVERSE, KERNEL_CHOICE KERNEL)
{
	assert(Msave <= M and GOT_MOMENTS == true and P==CHEBYSHEV);
	
	double a=this->a; double b=this->b; double Emin=this->Emin; double Emax=this->Emax;
	int Epoints = max(2*Msave,4096); // at least 4096 Epoints for smooth plots
	VectorXd ImAAgamma = this->fct(ImAAmoments.head(Msave),Epoints,REVERSE,KERNEL);
	
	IntervalIterator Eit(Emin-Eoffset,Emax-Eoffset,Epoints,ChebyshevAbscissa);
	for (Eit=Eit.begin(); Eit<Eit.end(); ++Eit)
	{
		double E_scaled = (Eit.value()+Eoffset-b)/a;
		double res = ImAAgamma(Eit.index())*OrthPoly<P>::w(E_scaled)/abs(a);
		Eit << res;
	}
	
	return Eit.get_data();
}

template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
void OrthPolyGreen<Hamiltonian,VectorType,P>::
inject_ImAAmoments (const VectorXd ImAAmoments_input)
{
	ImAAmoments = ImAAmoments_input;
	GOT_MOMENTS = true;
	M = ImAAmoments.rows();
}

template<typename Hamiltonian, typename VectorType, ORTHPOLY P>
void OrthPolyGreen<Hamiltonian,VectorType,P>::
predict_ImAA (int M_new)
{
	M += M_new;
	insert_linearPrediction(ImAAmoments,M_new);
}

#endif

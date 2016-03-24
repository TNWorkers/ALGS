#ifndef LINEAR_PREDICTION
#define LINEAR_PREDICTION

MatrixXd pseudoInv (const MatrixXd &Min, double delta=1e-6)
{
	JacobiSVD<MatrixXd> Jack(Min,ComputeFullU|ComputeFullV);
	VectorXd Sinv = Jack.singularValues();
	
	for (long i=0; i<Sinv.rows(); ++i)
	{
		if (Sinv(i) > delta)
		{
			Sinv(i) = 1./Sinv(i);
		}
		else
		{
			Sinv(i) = 0;
		}
	}
	return Jack.matrixV() * Sinv.asDiagonal() * Jack.matrixU().transpose();
}

VectorXd linearPrediction (const VectorXd &x, int N_new, double delta=0.)
{
	int L = x.rows()/2;
	
	MatrixXd R(L,L);
	R.setZero();
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	for (int n=L; n<x.rows(); ++n)
	{
		R(i,j) += x(n-i-1)*x(n-j-1);
	}
	
	VectorXd r(L);
	r.setZero();
	for (int i=0; i<L; ++i)
	for (int n=L; n<x.rows(); ++n)
	{
		r(i) -= x(n-i-1)*x(n);
	}
	
	VectorXd a;
//	if (delta==0.)
//	{
//		FullPivLU<MatrixXd> LinSolver(R);
//		a = LinSolver.solve(r);
		a = R.fullPivHouseholderQr().solve(r);
//		a = R.fullPivLu().solve(r);
//	}
//	//using pseudo-inverse; this is really bad for simple cosine points (???)
//	else
//	{
//		a = pseudoInv(R,1e-7)*r;
//	}
	
	// test a:
//	for (int n=x.rows()/2; n<x.rows(); ++n)
//	{
//		double xx = 0.;
//		for (int j=0; j<L; ++j)
//		{
//			xx -= a(j)*x(n-j-1);
//		}
//		cout << x(n) << "\t" << xx << endl;
//	}
	
	MatrixXd M(L,L);
	M.setZero();
	M.diagonal<-1>().setConstant(1.);
	M.row(0) = -a.transpose();
	
	EigenSolver<MatrixXd> Eugen(M);
	VectorXcd lambda = Eugen.eigenvalues();
	int N_div = 0;
	for (int i=0; i<lambda.rows(); ++i)
	{
		// deal with divergent eigenvalues:
		if (abs(lambda(i))>1)
		{
//			lambda(i) = 0;
//			lambda(i) = lambda(i)/abs(lambda(i));
			lambda(i) = 1./conj(lambda(i));
			++N_div;
		}
	}
	cout << "divergent eigenvalues: " << N_div << endl;
	VectorXcd xl = Eugen.eigenvectors().row(0);
	VectorXcd xr = Eugen.eigenvectors().inverse() * x.head(L).reverse();
	
	VectorXd Vout(N_new);
	for (int k=x.rows()-L; k<x.rows()-L+N_new; ++k)
	{
		complex<double> x_new = xl.transpose() * lambda.array().pow(k+1).matrix().asDiagonal() * xr;
		Vout(k-x.rows()+L) = x_new.real();
	}
	
	return Vout;
}

void insert_linearPrediction (VectorXd &x, int N_new, double delta=0.)
{
	VectorXd pred = linearPrediction(x,N_new,delta);
	x.conservativeResize(x.rows()+N_new);
	x.tail(N_new) = pred;
}

#endif

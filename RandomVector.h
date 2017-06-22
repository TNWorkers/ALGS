#ifndef RANDOMVECTOR
#define RANDOMVECTOR

//#include <boost/random/mersenne_twister.hpp>
//#include <boost/random/normal_distribution.hpp>
//#include <boost/random/uniform_real_distribution.hpp>
//#include <boost/random/variate_generator.hpp>

//boost::random::mt11213b MtEngine;
//boost::normal_distribution<double> NormDist(0.,1.); // mean=0, sigma=1
//boost::random::uniform_01<double> UniformDist; // lower=0, upper=1
//boost::random::uniform_real_distribution<double> UniformDist0; // lower=-1, upper=1

#include <random>
#include <thread>

template<typename Scalar> Scalar threadSafeRandUniform (double min, double max, bool SEED=false) {};
template<typename Scalar> Scalar threadSafeRandNormal (double mean, double sigma) {};

template<>
double threadSafeRandUniform<double> (double min, double max, bool SEED)
{
	static thread_local mt19937 generatorUniformReal(random_device{}());
	if ( SEED ) { generatorUniformReal.seed(1); }
	uniform_real_distribution<double> distribution(min, max);
	return distribution(generatorUniformReal);
}

template<>
complex<double> threadSafeRandUniform<complex<double> > (double min, double max, bool SEED)
{
	static thread_local mt19937 generatorUniformComplex(random_device{}());
	if ( SEED ) { generatorUniformComplex.seed(1); }
	uniform_real_distribution<double> distribution(min, max);
	return complex<double>(distribution(generatorUniformComplex),distribution(generatorUniformComplex));
}

template<>
double threadSafeRandNormal<double> (double mean, double sigma)
{
	static thread_local mt19937 generatorNormalReal(random_device{}());
	normal_distribution<double> distribution(mean, sigma);
	return distribution(generatorNormalReal);
}

template<>
complex<double> threadSafeRandNormal<complex<double> > (double mean, double sigma)
{
	static thread_local mt19937 generatorNormalComplex(random_device{}());
	normal_distribution<double> distribution(mean, sigma);
	return complex<double>(distribution(generatorNormalComplex),distribution(generatorNormalComplex));
}

//template<typename VectorType, typename Scalar>
//struct GaussianRandomVector
//{
//	static void fill (size_t N, VectorType &Vout);
//};

template<typename VectorType, typename Scalar>
struct GaussianRandomVector
{
	static void fill (size_t N, VectorType &Vout)
	{
		Vout.resize(N);
		for (size_t i=0; i<N; ++i) {Vout(i) = threadSafeRandUniform<Scalar>(-1.,1.);}
		normalize(Vout);
	}
};

//template<typename VectorType>
//struct GaussianRandomVector<VectorType,complex<double> >
//{
//	static void fill (size_t N, VectorType &Vout)
//	{
//		Vout.resize(N);
//		for (size_t i=0; i<N; ++i) {Vout(i) = complex<double>(threadSafeRandNormal(0.,1.), threadSafeRandNormal(0.,1.));}
//		normalize(Vout);
//	}
//};

MatrixXd randOrtho (size_t N)
{
	MatrixXd M(N,N);
	for (size_t i=0; i<N; ++i)
	for (size_t j=0; j<N; ++j)
	{
		M(i,j) = threadSafeRandUniform<double>(0.,1.);
	}
	HouseholderQR<MatrixXd> Quirinus(M);
	MatrixXd Qmatrix = MatrixXd::Identity(N,N);
	Qmatrix = Quirinus.householderQ() * Qmatrix;
	return Qmatrix;
}

#endif

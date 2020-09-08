#ifndef RANDOMVECTOR
#define RANDOMVECTOR

#include <Eigen/Dense>

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

template<typename Scalar, typename RealScalar> Scalar threadSafeRandUniform (RealScalar min, RealScalar max, bool SEED=false) {};
template<typename Scalar, typename RealScalar> Scalar threadSafeRandNormal (RealScalar mean, RealScalar sigma) {};

template<>
double threadSafeRandUniform<double,double> (double min, double max, bool SEED)
{
	static thread_local mt19937 generatorUniformReal(random_device{}());
	if (SEED) generatorUniformReal.seed(std::time(0));
	uniform_real_distribution<double> distribution(min, max);
	return distribution(generatorUniformReal);
}

template<>
complex<double> threadSafeRandUniform<complex<double>, double> (double min, double max, bool FIXED_SEED)
{
	static thread_local mt19937 generatorUniformComplex(random_device{}());
	if (FIXED_SEED) generatorUniformComplex.seed(std::time(0));
	uniform_real_distribution<double> distribution(min, max);
	return complex<double>(distribution(generatorUniformComplex), distribution(generatorUniformComplex));
}

template<>
int threadSafeRandUniform<int,int> (int min, int max, bool SEED)
{
	static thread_local mt19937 generatorUniformInt(random_device{}());
	if (SEED) generatorUniformInt.seed(std::time(0));
	uniform_int_distribution<unsigned> distribution(min, max);
	return distribution(generatorUniformInt);
}

template<>
double threadSafeRandNormal<double,double> (double mean, double sigma)
{
	static thread_local mt19937 generatorNormalReal(random_device{}());
	normal_distribution<double> distribution(mean, sigma);
	return distribution(generatorNormalReal);
}

template<>
complex<double> threadSafeRandNormal<complex<double>,double> (double mean, double sigma)
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
		for (size_t i=0; i<N; ++i) {Vout(i) = threadSafeRandUniform<Scalar,double>(-1.,1.);}
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

Eigen::MatrixXd randOrtho (size_t N)
{
	Eigen::MatrixXd M(N,N);
	for (size_t i=0; i<N; ++i)
	for (size_t j=0; j<N; ++j)
	{
		M(i,j) = threadSafeRandUniform<double,double>(0.,1.);
	}
	Eigen::HouseholderQR<Eigen::MatrixXd> Quirinus(M);
	Eigen::MatrixXd Qmatrix = Eigen::MatrixXd::Identity(N,N);
	Qmatrix = Quirinus.householderQ() * Qmatrix;
	return Qmatrix;
}

#endif

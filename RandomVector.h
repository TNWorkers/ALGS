#ifndef RANDOMVECTOR
#define RANDOMVECTOR

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>

boost::random::mt11213b MtEngine;
boost::normal_distribution<double> NormDist(0.,1.); // mean=0, sigma=1
boost::random::uniform_01<double> UniformDist; // lower=0, upper=1
boost::random::uniform_real_distribution<double> UniformDist0;

//typedef boost::mt19937 MersenneTwister; // Mersenne twister
//typedef boost::normal_distribution<double> NormDist; // normal distribution
//typedef boost::variate_generator<MersenneTwister,NormDist> RanGen; // variate generator
//MersenneTwister RanEngine;
//NormDist      GlobalNormDist(0.,1.); // mean=0, sigma=1
//Uniform01Dist GlobalUniform01Dist;
//RanGen   ran  (RanEngine,GlobalNormDist);

//template<typename VectorType>
//void GaussianRandomVector (size_t N, VectorType &Vout)
//{
//	Vout.resize(N);
//	for (size_t i=0; i<N; ++i) {Vout(i) = NormDist(MtEngine);}
//	Vout /= norm(Vout);
//}

template<typename VectorType, typename Scalar>
struct GaussianRandomVector
{
	static void fill (size_t N, VectorType &Vout);
};

template<typename VectorType>
struct GaussianRandomVector<VectorType,double>
{
	static void fill (size_t N, VectorType &Vout)
	{
		Vout.resize(N);
		for (size_t i=0; i<N; ++i) {Vout(i) = NormDist(MtEngine);}
		normalize(Vout);
	}
};

template<typename VectorType>
struct GaussianRandomVector<VectorType,complex<double> >
{
	static void fill (size_t N, VectorType &Vout)
	{
		Vout.resize(N);
		for (size_t i=0; i<N; ++i) {Vout(i) = complex<double>(NormDist(MtEngine), NormDist(MtEngine));}
		normalize(Vout);
	}
};

MatrixXd randOrtho (size_t N)
{
	MatrixXd M(N,N);
	for (size_t i=0; i<N; ++i)
	for (size_t j=0; j<N; ++j)
	{
		M(i,j) = UniformDist(MtEngine);
	}
	HouseholderQR<MatrixXd> Quirinus(M);
	MatrixXd Qmatrix = MatrixXd::Identity(N,N);
	Qmatrix = Quirinus.householderQ() * Qmatrix;
	return Qmatrix;
}

#endif

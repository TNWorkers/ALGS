#ifndef BZ_SUM_SOLVER_H_
#define BZ_SUM_SOLVER_H_

#include <vector>
#include <set>
#include <numeric>
#include <cmath>
#include <fstream>
#include <execution>

#include <boost/rational.hpp>

#include "Eigen/Dense"

#include "termcolor/termcolor.hpp"

#include "HDF5Interface.h"

template<typename Scalar>
struct MatrixComparator33 {
    bool operator() (const Eigen::Matrix<Scalar,3,3>& lhs, const Eigen::Matrix<Scalar,3,3>& rhs) const
        {
            return std::tie(lhs(0,0), lhs(1,0), lhs(2,0), lhs(0,1), lhs(1,1), lhs(2,1), lhs(0,2), lhs(1,2), lhs(2,2)) <
                std::tie(rhs(0,0), rhs(1,0), rhs(2,0), rhs(0,1), rhs(1,1), rhs(2,1), rhs(0,2), rhs(1,2), rhs(2,2));
        }
};

enum class LatticeType {
    fcc,
    sc ,
    bcc
};

/** \class BZSumSolver
  *
  * This solver can compute k-summations over the 1st Brillouin-Zone by generating a suitable subset of k-points.
  * Each k-point is associated with a weight which accounts for the number of symmetry-equivalent k-points.
  * The algorithm follows the idea of Chadi & Cohen (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.8.5747).
  * This algorithm is quite old and there are probably better ones. For example: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.93.155109.
  * Feel free to implement!
  *
  * Currently, supported lattices are sc, bcc and fcc.
  *
  * K-points sets can be saved to hdf5 via save(). This is useful for higher orders since the generation of the set can be time consuming.
  * \note This class is hard-coded for 3D. It can be easily customized for other dimensions.
  */
template<typename T, typename Scalar>
class BZSumSolver {
    typedef Eigen::Matrix<Scalar, 3, 1> Kpoint;
    typedef Eigen::Matrix<int, 3, 3> SymOp;
    
public:
    BZSumSolver(const int order_in, const LatticeType lattice_in) :
        order(order_in), lattice(lattice_in) { init(); gen_kpoints(); }

    explicit BZSumSolver(const std::string& filename);
    
    T sum(const std::function<T(Kpoint)>& k_func, const T& init) const;

    Scalar inverseNorm() const;

    void checkGroupStructure() const;

    void save(const std::string& filename) const;

    std::string info() const;
private:
    void init();
    void gen_kpoints();
    void set_Oh_generators();
    
    int order;
    LatticeType lattice;
    
    std::vector<Kpoint> k_points;
    std::vector<int> weights;
    int weight_sum;
    Kpoint k1;
    Kpoint k2;

    std::set<SymOp, MatrixComparator33<int>> point_group;
    std::vector<SymOp> generators_point_group;
    Eigen::Matrix<Scalar,3,3> reciprocal_basis;
    Eigen::Matrix<Scalar,3,3> inv_reciprocal_basis;
};

template<typename T, typename Scalar>
BZSumSolver<T,Scalar>::BZSumSolver(const std::string& filename)
{
    HDF5Interface source(filename, FILE_ACCESS_MODE::READ);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> all_ks;
    source.load_matrix(all_ks, "k_points");
    k_points.resize(all_ks.cols());
    for (std::size_t i=0; i<all_ks.cols(); i++) {k_points[i] = all_ks.col(i);}
    weights.resize(k_points.size());
    source.load_vector<int>(weights.data(), "weights"); 
    weight_sum = std::accumulate(weights.begin(), weights.end(), 0);
    std::cout << info() << std::endl;
    // std::size_t count_k=0;
    // for (const auto& k:k_points) {
    //     boost::rational<int> w(weights[count_k], weight_sum);
    //     std::cout << std::fixed << k.transpose() << " " << w << std::endl;
    //     count_k++;
    // }
}

template<typename T, typename Scalar>
std::string BZSumSolver<T,Scalar>::info() const
{
    std::stringstream ss;
    ss << termcolor::colorize << termcolor::bold << "BZSumSolver:" << termcolor::reset << " # k-points=" << k_points.size() << ", # weights=" << weights.size() << ", Σ(weights)=" << weight_sum;
    return ss.str();
}

template<typename T, typename Scalar>
void BZSumSolver<T,Scalar>::set_Oh_generators()
{
    SymOp tmp; tmp.setZero();
    generators_point_group.push_back(SymOp::Identity());
    tmp.diagonal() << -1, 1, 1;
    generators_point_group.push_back(tmp);
    tmp.diagonal() << -1, 1, -1;
    generators_point_group.push_back(tmp);
    tmp.diagonal() << -1, -1, -1;
    generators_point_group.push_back(tmp);
    tmp.setZero();
    tmp << 0, 0, 1, 1, 0, 0, 0, 1 , 0;
    generators_point_group.push_back(tmp);
    tmp.setZero();
    tmp << 0, 1, 0, 1, 0, 0, 0, 0 , -1;
    generators_point_group.push_back(tmp);
}

template<typename T, typename Scalar>
void BZSumSolver<T,Scalar>::init()
{
    switch (lattice) {
    case LatticeType::fcc :
        //initial points are taken from the Chadi&Cohen paper.
        k1 << 0.5, 0.5, 0.5;
        k2 << 0.25, 0.25, 0.25;
        reciprocal_basis <<
             1.,  1., -1.,
             1., -1.,  1.,
            -1.,  1.,  1.;
        set_Oh_generators();
        break;
    case LatticeType::sc :
        //initial points are taken from the Chadi&Cohen paper.
        k1 << 0.25, 0.25, 0.25;
        k2 << 0.125, 0.125, 0.125;
        reciprocal_basis <<
            1., 0., 0.,
            0., 1., 0.,
            0., 0., 1.;
        set_Oh_generators();
        break;
    case LatticeType::bcc :
        //initial points are taken from the Chadi&Cohen paper.
        k1 << 0.5, 0.5, 0.5;
        k2 << 0.25, 0.25, 0.25;
        reciprocal_basis <<
            1., 1., 0.,
            1., 0., 1.,
            0., 1., 1.;
        set_Oh_generators();
        break;

    }
    inv_reciprocal_basis = reciprocal_basis.inverse();

    //Generate the whole group by combining the generators.
    //Here, we hope that the product of 4 generators is enough.
    //Enhance the number if you do not get the full group. This step is not time critical.
    for (const auto& m:generators_point_group) {
        point_group.insert(m);
        for (const auto& n:generators_point_group) {
            point_group.insert(m*n);
            for (const auto& k:generators_point_group) {
                point_group.insert(m*n*k);
                for (const auto& l:generators_point_group) {
                    point_group.insert(m*n*k*l);
                }
            }
        }
    }

    // std::cout << "# elements in group: " << point_group.size() << std::endl;
}

template<typename T, typename Scalar>
void BZSumSolver<T,Scalar>::save(const std::string& filename) const
{
    HDF5Interface target(filename, FILE_ACCESS_MODE::WRITE);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> all_ks; all_ks.resize(3, k_points.size());
    std::size_t count=0; for (const auto& k:k_points) {all_ks.col(count) = k; count++;}
    target.save_matrix(all_ks, "k_points");
    target.save_vector<int>(weights.data(), weights.size(), "weights");
}

template<typename T, typename Scalar>
void BZSumSolver<T,Scalar>::gen_kpoints()
{
    std::vector<Kpoint> k_points_tmp;
    std::vector<int> weights_tmp;
    k_points.push_back(k1);
    weights.push_back(1);

    for (std::size_t i=1; i<order; i++)
    {
        // std::cout << "Computing order=" << i << std::endl;
        Kpoint seed = 2*k1(0)/std::pow(2,i) * k1;
        for (const auto& k:k_points) {
            for (auto& Op:point_group) {
                Kpoint candidate = k + Op.cast<Scalar>()*seed;
                //This comparison checks if to k-points are related by a symmetry transformation of the point group or by a reciprocal lattice vector.
                auto comp = [this, &candidate] (const Kpoint& k_point) -> bool {
                    for (const auto& m:point_group) {
                        Eigen::Matrix<Scalar,3,1> int_check = inv_reciprocal_basis*(candidate-m.template cast<Scalar>()*k_point);
                        if ((int_check - int_check.array().round().matrix()).norm() < 1.e-8) {
                            return true;
                        }
                    }
                    return false;
                };
                auto it = std::find_if(std::execution::par_unseq, k_points_tmp.begin(), k_points_tmp.end(), comp);
                if (it == k_points_tmp.end()) {
                    k_points_tmp.push_back(candidate);
                    weights_tmp.push_back(1);
                }
                else { weights_tmp[std::distance(k_points_tmp.begin(),it)] += 1; }
            }
        }
        k_points = k_points_tmp;
        weights = weights_tmp;
        weights_tmp.clear();
        k_points_tmp.clear();
    }
    weight_sum = std::accumulate(weights.begin(), weights.end(), 0);
    std::cout << info() << std::endl;
    // std::size_t count_k=0;
    // for (const auto& k:k_points) {
    //     boost::rational<int> w(weights[count_k], weight_sum);
    //     std::cout << std::fixed << k.transpose() << " " << w << std::endl;
    //     count_k++;
    // }
}

template<typename T, typename Scalar>
T BZSumSolver<T,Scalar>::sum(const std::function<T(Kpoint)>& k_func, const T& init) const {
    T out = init;
    for (std::size_t ik=0; ik<k_points.size(); ik++) {
        out += k_func(2*M_PI*k_points[ik]) * (static_cast<Scalar>(weights[ik]) / static_cast<Scalar>(weight_sum));
    }
    return out;
}

template<typename T, typename Scalar>
Scalar BZSumSolver<T,Scalar>::inverseNorm() const {
    auto k_func = [] (const Kpoint& k) -> Scalar {return 1.;};
    return 1./sum(k_func,0.);
}

template<typename T, typename Scalar>
void BZSumSolver<T,Scalar>::checkGroupStructure() const {
     for (const auto& m:point_group) {
        std::cout << "element:" << std::endl << std::fixed << m << ", det(m)=" << m.determinant();
        if ((m-SymOp::Identity()).norm() == 0) {std::cout << ", order=1" << std::endl;}
        else if ((m*m-SymOp::Identity()).norm() == 0) {std::cout << ", order=2" << std::endl;}
        else if ((m*m*m-SymOp::Identity()).norm() == 0) {std::cout << ", order=3" << std::endl;}
        else if ((m*m*m*m-SymOp::Identity()).norm() == 0) {std::cout << ", order=4" << std::endl;}
        else if ((m*m*m*m*m-SymOp::Identity()).norm() == 0) {std::cout << ", order=5" << std::endl;}
        else if ((m*m*m*m*m*m-SymOp::Identity()).norm() == 0) {std::cout << ", order=6" << std::endl;}
        else {std::cout << ", order>6" << std::endl;}
    }
    std::vector<bool> INVERSE_EXISTS(48, false);
    std::size_t count_m=0;
    for (const auto& m:point_group) {
        for (const auto& n:point_group) {
            if ((m*n - SymOp::Identity()).norm() == 0) {INVERSE_EXISTS[count_m] = true;}
        }
        count_m++;
    }
    std::size_t count_n=0;
    count_m=0;
    std::vector<std::vector<bool>> IS_CLOSED(48, INVERSE_EXISTS);
    for (const auto& m:point_group) {
        for (const auto& n:point_group) {
            for (const auto& l:point_group) {
                if ((m*n - l).norm() == 0) {IS_CLOSED[count_m][count_n] = true;}
            }
            count_n++;
        }
        count_m++;
    }
    std::cout << "group axioms: closed?" << std::endl;
    for (const auto& tmp:IS_CLOSED) {
        for (const auto b:tmp ) {
            std::cout << std::boolalpha << b << std::endl;
        }
    }
    std::cout << "inverse?" << std::endl;
    for (const auto b:INVERSE_EXISTS) {
        std::cout << std::boolalpha << b << std::endl;
    }
}
#endif

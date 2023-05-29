#include <iostream>

#include <Eigen/Dense>
#include "NewtonMethod.hpp"

// example 3.7
Eigen::VectorXd example3_7(const Eigen::VectorXd& p) {
    Eigen::VectorXd ret = Eigen::VectorXd::Zero(p.rows());
    ret(0) = 2.0*p(0)*p(0)*p(0)+4.0*p(0)*p(0)+p(0)-2.0;
    return ret;
}

Eigen::MatrixXd example3_7_jacob(const Eigen::VectorXd& p) {
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(p.rows(), p.rows());
    ret(0, 0) = 6.0 * p(0) * p(0) + 8.0 * p(0) + 1.0;
    return ret;
}

// example 3.8
Eigen::VectorXd example3_8(const Eigen::VectorXd& p) {
    Eigen::VectorXd ret = Eigen::VectorXd::Zero(p.rows());
    ret(0) = p(1) - 1.0/p(0);
    ret(1) = p(1) - sqrt(p(0));
    return ret;
}

Eigen::MatrixXd example3_8_jacob(const Eigen::VectorXd& p) {
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(p.rows(), p.rows());
    ret(0, 0) = 1.0 / p(0) / p(0);
    ret(0, 1) = 1.0;
    ret(1, 0) = -0.5/sqrt(p(0));
    ret(1, 1) = 1.0;
    return ret;
}

// example 
int main() {
    NewtonMethod nm1(1, example3_7, example3_7_jacob);
    Eigen::VectorXd sol = nm1.solve(1.5*Eigen::VectorXd::Ones(1));
    std::cout << sol << std::endl;
    sol = nm1.solve(-0.5*Eigen::VectorXd::Ones(1));
    std::cout << sol << std::endl;

    NewtonMethod nm2(2, example3_8, example3_8_jacob);
    Eigen::VectorXd initial_point = Eigen::VectorXd::Zero(2);
    initial_point << 2.0, 3.0;
    sol = nm2.solve(initial_point);
    std::cout << sol << std::endl;

}
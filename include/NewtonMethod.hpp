#pragma once

#include <iostream>

#include <Eigen/Dense>

// solve non-linear equation by Newton's method.
// use analytic derivative if provided
// template parameter N is the dimension of input variable
// i.e. given f: R^N -> R^N, find x s.t. f(x) = 0
class NewtonMethod {
public:
    NewtonMethod(int N, Eigen::VectorXd (*target_func)(const Eigen::VectorXd&), Eigen::MatrixXd (*jacob)(const Eigen::VectorXd&));
    Eigen::VectorXd solve(Eigen::VectorXd initial_point, double tolerance = 1e-9, int max_iteration = -1);

private:
    const int N;
    Eigen::VectorXd (*target_func)(const Eigen::VectorXd&);
    Eigen::MatrixXd (*get_jacobian)(const Eigen::VectorXd&);
};
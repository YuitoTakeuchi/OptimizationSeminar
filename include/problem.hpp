#pragma once
#include <Eigen/Core>

// 最適化問題
// min.   f(x, u)
// w.r.t. x
// s.t.   g(x, u) <= 0
//        h(x, u)  = 0
//        xl <= x <= xu
//        r(x, u) = 0      // residual equation. u is root of this(these) equation.

// N: the number of inequality constraints
// M: the number of   equality constraints
template<int N, int M>
class OptimizationProblem {
private:
    double (*objective_func)(Eigen::VectorXd);
    double (*g[N])(Eigen::VectorXd);
    double (*h[M])(Eigen::VectorXd);
    double (*r)(Eigen::VectorXd); // residual equation

    Eigen::VectorXd x;
    Eigen::VectorXd xl, xu;

public:
    OptimizationProblem(){};

};
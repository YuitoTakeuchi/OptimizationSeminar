// #include "SearchDirection.hpp"
// #include "LineSearch.hpp"

#include "NonlinearEquationSolver.hpp"
#include <filesystem>

constexpr double beta = 1.0;

double lagrangian(const Eigen::VectorXd& x) {
    double x1 = x(0), x2 = x(1), lambda = x(2);

    return x1*x1 + 3.0*(x2-2.0)*(x2-2.0) + lambda * (beta*x1*x1 - x2);
}

Eigen::VectorXd grad(const Eigen::VectorXd& x) {
    double x1 = x(0), x2 = x(1), lambda = x(2);
    Eigen::VectorXd ret = x;
    ret(0) = 2.0 * x1 * (1.0 + lambda * beta);
    ret(1) = 6.0 * (x2-2.0) - lambda;
    ret(2) = beta * x1*x1 - x2;
    return ret;
}

Eigen::MatrixXd hessian(const Eigen::VectorXd& x) {
    double x1 = x(0), x2 = x(1), lambda = x(2);
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(3, 3);

    ret(0, 0) = 2.0*(1.0+lambda*beta); ret(0, 1) = 0.0; ret(0, 2) = 2.0*x1*beta;
    ret(1, 0) = 0.0; ret(1, 1) = 6.0; ret(1, 2) = -1.0;
    ret(2, 0) = 2.0*beta*x1; ret(2, 1) = -1.0; ret(2, 2) = 0.0;
    return ret;
}


int main() {
    // Newton法でgrad = 0を解く
    NLEqSolver::NewtonMethod solver(3, &grad, &hessian);
    Eigen::VectorXd x = Eigen::VectorXd::Ones(3);
    solver.solve(x, 1e-6, 100);
}
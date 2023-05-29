#include "NewtonMethod.hpp"
#include <filesystem>

Eigen::VectorXd grad(const Eigen::VectorXd& x) {
    double x1 = x(0), x2 = x(1), sigma = x(2), s = x(3);
    Eigen::VectorXd ret = x;
    ret(0) = 1.0 + 0.5 * sigma * x1;
    ret(1) = 2.0 + 2.0 * sigma * x2;
    ret(2) = 0.25 * x1 * x1 + x2 * x2 - 1.0;
    ret(3) = 2.0 * sigma * s;
    return ret;
}

Eigen::MatrixXd hessian(const Eigen::VectorXd& x) {
    double x1 = x(0), x2 = x(1), sigma = x(2), s = x(3);
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(4, 4);
    ret << 0.5*sigma,       0.0, 0.5*x1,       0.0,
                 0.0, 2.0*sigma, 2.0*x2,       0.0,
              0.5*x1,    2.0*x2,    0.0,       0.0,
                 0.0,       0.0,  2.0*s, 2.0*sigma;
    return ret;
}


int main() {
    // Newton法でgrad = 0を解く
    NewtonMethod solver(4, &grad, &hessian);
    Eigen::VectorXd x = Eigen::VectorXd::Ones(4);
    solver.solve(x, 1e-6, 100);
}
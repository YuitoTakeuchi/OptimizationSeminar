#include "SearchDirection.hpp"
#include "LineSearch.hpp"

double func(const Eigen::VectorXd& point) {
    double ret = 0.0;
    double x1 = point(0), x2 = point(1);
    ret += (1.0-x1)*(1.0-x1) + (1.0-x2)*(1.0-x2) + 0.5*(2.0*x2-x1*x1)*(2.0*x2-x1*x1);
    return ret;
}

Eigen::VectorXd calc_grad(const Eigen::VectorXd& point) {
    Eigen::VectorXd ret = Eigen::VectorXd::Zero(point.rows());
    double x1 = point(0), x2 = point(1);
    ret(0) = -2+2.0*x1-4.0*x1*x2+2.0*x1*x1*x1;
    ret(1) = -2.0+6.0*x2-2.0*x1*x1;
    return ret;
}

Eigen::MatrixXd calc_hessian(const Eigen::VectorXd& x) {
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(2, 2);
    double x1 = x(0), x2 = x(1);
    ret(0, 0) = 2.0-4.0*x2+6.0*x1*x1;
    ret(0, 1) = -4.0*x1;
    ret(1, 0) = -4.0*x1;
    ret(1, 1) = 6.0;
    return ret;
}

int main() {
    NewtonsMethod<Armijo> gd(&func, &calc_grad, &calc_hessian);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
    x << -1, 2;
    gd.solve(x, 1e-6);
    x = gd.get_optimal_point();
    double J = gd.get_optimal_value();
    std::cout << "optimaized: (x1, x2) = (" << x(0) << ", " << x(1) << ")\n";
    std::cout << "J = " << J << "\n";
}
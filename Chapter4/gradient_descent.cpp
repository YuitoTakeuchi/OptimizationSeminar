#include "SearchDirection.hpp"
#include "LineSearch.hpp"

double func(const Eigen::VectorXd& point) {
    double ret = 0.0;
    double x1 = point(0), x2 = point(1);
    ret += 0.1 * pow(x1, 6) - 1.5 * pow(x1, 4) + 0.2*pow(x2, 4) + 3.0*pow(x2, 2) - 9.0*x2 + 0.5*x1*x2;
    return ret;
}

Eigen::VectorXd calc_grad(const Eigen::VectorXd& point) {
    Eigen::VectorXd ret = Eigen::VectorXd::Zero(point.rows());
    double x1 = point(0), x2 = point(1);
    ret(0) = 0.6*pow(x1, 5) - 6.0 * pow(x1, 3) + 0.5*x2;
    ret(1) = 0.8*pow(x2, 3) + 6.0 * x2 - 9.0 + 0.5*x1;
    return ret;
}

int main() {
    GradientDescent<Armijo> gd(&func, &calc_grad);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
    x << -1, 1;
    gd.solve(x, 1e-6);
    x = gd.get_optimal_point();
    double J = gd.get_optimal_value();
    std::cout << "optimaized: (x1, x2) = (" << x(0) << ", " << x(1) << ")\n";
    std::cout << "J = " << J << "\n";
}
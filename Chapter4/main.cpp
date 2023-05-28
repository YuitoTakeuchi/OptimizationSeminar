// implementation of algorithm 4-1
// line search algorithm

#include <Eigen/Core>
#include <iostream>

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

Eigen::VectorXd gradient_descent(Eigen::VectorXd point) {
    return -calc_grad(point);
}

void line_search(Eigen::VectorXd& point, double tolerance = 1e-9) {
    Eigen::VectorXd grad = Eigen::VectorXd::Ones(point.rows());
    while(grad.maxCoeff() > tolerance || -grad.minCoeff() > tolerance) {
        Eigen::VectorXd search_direction = gradient_descent(point);
        LineSearch ls(&func, &calc_grad, func(point), calc_grad(point).dot(search_direction), point, search_direction);
        point += ls.find_step() * search_direction;
        grad = calc_grad(point);
    }
}

int main() {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
    x << -1.0, 1.0;
    line_search(x, 1e-6);

    std::cout << x << std::endl;
}
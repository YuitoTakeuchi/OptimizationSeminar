#include "SearchDirection.hpp"
#include "LineSearch.hpp"
#include <filesystem>

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
    std::filesystem::path p = __FILE__;

    Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
    x << -1, 2;

    GradientDescent<Armijo> gradient_descent(2, &func, &calc_grad);
    gradient_descent.solve(x, 1e-6);
    gradient_descent.output_to_file(std::string(p.parent_path()) + "/result/gradient_descent.txt");
    
    ConjugateGradient<Armijo> conjugate_gradient(2, &func, &calc_grad);
    conjugate_gradient.solve(x, 1e-6);
    conjugate_gradient.output_to_file(std::string(p.parent_path()) + "/result/conjugate_gradient.txt");

    NewtonMethod<Armijo> newtons_method(2, &func, &calc_grad, &calc_hessian);
    newtons_method.solve(x, 1e-6);
    newtons_method.output_to_file(std::string(p.parent_path()) + "/result/newtons_method.txt");

    BFGS<Armijo> BFGS(2, &func, &calc_grad);
    BFGS.solve(x, 1e-6);
    BFGS.output_to_file(std::string(p.parent_path()) + "/result/BFGS.txt");
}
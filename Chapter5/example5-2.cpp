#include "SearchDirection.hpp"
#include "LineSearch.hpp"
#include <filesystem>

// ラグランジュの未定乗数法で解く
// min.   x_1 + 2x_2
// w.r.t. x_1, x_2
// s.t.   1/4 x_1^2 + x_2^2 -1 = 0

double lagrangian(const Eigen::VectorXd& x) {
    double x1 = x(0), x2 = x(1), lambda = x(2);

    return x1 + 2.0*x2 + lambda * (0.25*x1*x1 + x2*x2 - 1.0);
}

Eigen::VectorXd grad(const Eigen::VectorXd& x) {
    double x1 = x(0), x2 = x(1), lambda = x(2);
    Eigen::VectorXd ret = x;
    ret(0) = 1.0 + 0.5 * lambda * x1;
    ret(1) = 2.0 + 2.0 * lambda * x2;
    ret(2) = 0.25 * x1*x1 + x2*x2 - 1.0;
    return ret;
}

int main() {
    // これだと発散してうまく行かない
    BFGS<Armijo> bfgs_solver(&lagrangian, &grad);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(3); x << -sqrt(2), sqrt(2)/2.0, sqrt(2)+0.01;
    bfgs_solver.solve(x, 1e-6);

    std::filesystem::path p = __FILE__;
    bfgs_solver.output_to_file(std::string(p.parent_path()) + "/result/BFGS.txt");
}
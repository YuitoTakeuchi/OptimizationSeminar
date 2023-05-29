#include "SearchDirection.hpp"
#include "LineSearch.hpp"
#include <filesystem>

// ラグランジュの未定乗数法で解く

constexpr double beta = -0.5;

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

int main() {
    // 発散する
    BFGS<Armijo> bfgs_solver(&lagrangian, &grad, 1000);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(3); x << 1.0, 1.0, 1.0;
    bfgs_solver.solve(x, 1e-6);

    std::filesystem::path p = __FILE__;
    bfgs_solver.output_to_file(std::string(p.parent_path()) + "/result/BFGS.txt");
}
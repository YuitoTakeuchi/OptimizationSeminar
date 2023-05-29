#include "SearchDirection.hpp"
#include "LineSearch.hpp"

#include <fstream>
#include <filesystem>
#include <iomanip>

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
    ConjugateGradient<Armijo> gd(&func, &calc_grad, 10);
    gd.set_store_points(true);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
    x << -1, 1;
    gd.solve(x, 1e-6);
    int cnt = gd.get_iter();
    if(cnt < 0) {
        std::cout << "solver did not converged\n";
        std::exit(EXIT_FAILURE);
    }
    x = gd.get_optimal_point();
    double J = gd.get_optimal_value();
    std::cout << "optimaized: (x1, x2) = (" << x(0) << ", " << x(1) << ")\n";
    std::cout << "J = " << J << "\n";

    std::cout << gd.get_iter() << " loop to converge\n";
    std::filesystem::path p = __FILE__;

    std::ofstream ofs(std::string(p.parent_path()) + "/result/res.txt");
    ofs << std::setprecision(15);
    for(int i = 0; i < gd.get_iter(); ++i) {
        for(int j = 0; j < 2; ++j) ofs << gd.get_point_history()[i](j) << " ";
        ofs << gd.get_obj_hisotry()[i] << "\n";
    }
    ofs.close();
}
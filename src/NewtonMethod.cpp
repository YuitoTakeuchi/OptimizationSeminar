#include "NewtonMethod.hpp"

NewtonMethod::NewtonMethod(int N, Eigen::VectorXd (*target_func)(Eigen::VectorXd), Eigen::MatrixXd (*jacob)(Eigen::VectorXd))
: N(N), target_func(target_func), get_jacobian(jacob) {

}

Eigen::VectorXd NewtonMethod::solve(Eigen::VectorXd initial_point, double tolerance, int max_iteration) {
    Eigen::VectorXd estimation = initial_point;
    Eigen::VectorXd current = Eigen::VectorXd::Ones(N);
    int cnt = 0;
    while(current.norm() > tolerance) {
        Eigen::MatrixXd jacob = get_jacobian(estimation);
        current = target_func(estimation);

        Eigen::FullPivLU<Eigen::MatrixXd> sol(jacob);
        Eigen::VectorXd delta = sol.solve(-current);
        estimation += delta;
        if(cnt++ > max_iteration && max_iteration > 0) {
            std::cout << "newton's method did not converge\n";
            break;
        }
    }
    return estimation;
}
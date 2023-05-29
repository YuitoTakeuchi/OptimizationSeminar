#include "NewtonMethod.hpp"

NewtonMethod::NewtonMethod(int N, Eigen::VectorXd (*target_func)(const Eigen::VectorXd&), Eigen::MatrixXd (*jacob)(const Eigen::VectorXd&))
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
            cnt = -1;
            break;
        }
    }

    std::cout << "\n\n*********************************************\n\n";
    if(cnt < 0) {
        std::cout << "Newton's Method Did NOT Converge.\n";
    } else {
        std::cout << "Nonlinear equation solved!\n";
        std::cout << "Solver: Newton's method" << "\n";
        std::cout << "Iteration Count: " << cnt << "\n";
        std::cout << "x = (";
        for(int i = 0; i < N; ++i) {
            std::cout << estimation(i) << (i == N-1 ? "" : ", ");
        }
        std::cout << ")\n";
    }
    std::cout << "\n\n*********************************************\n\n";
    return estimation;
}
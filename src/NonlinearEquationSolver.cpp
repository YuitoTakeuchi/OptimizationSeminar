#include "NonlinearEquationSolver.hpp"

NLEqSolver::NewtonMethod::NewtonMethod(int N, Eigen::VectorXd (*target_func)(const Eigen::VectorXd&), Eigen::MatrixXd (*jacob)(const Eigen::VectorXd&))
: N(N), target_func(target_func), get_jacobian(jacob) {

}

Eigen::VectorXd NLEqSolver::NewtonMethod::solve(Eigen::VectorXd initial_point, double tolerance, int max_iter) {
    Eigen::VectorXd estimation = initial_point;
    Eigen::VectorXd current = Eigen::VectorXd::Ones(N);
    int cnt = 0;
    while(current.norm() > tolerance) {
        Eigen::MatrixXd jacob = get_jacobian(estimation);
        current = target_func(estimation);

        Eigen::FullPivLU<Eigen::MatrixXd> sol(jacob);
        Eigen::VectorXd delta = sol.solve(-current);
        estimation += delta;
        if(cnt++ > max_iter && max_iter > 0) {
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

NLEqSolver::BFGS::BFGS(int N, Eigen::VectorXd (*target_func)(const Eigen::VectorXd&))
: N(N), target_func(target_func) {
    H = Eigen::MatrixXd::Identity(N, N);
}

Eigen::VectorXd NLEqSolver::BFGS::solve(Eigen::VectorXd initial_point, double tolerance, int max_iter) {
    Eigen::VectorXd estimation = initial_point;
    Eigen::VectorXd current;
    Eigen::VectorXd prev_x = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd prev_val = Eigen::VectorXd::Zero(N);

    int cnt = 0;
    while(cnt < max_iter) {
        // Hの更新
        current = target_func(estimation);
        std::cout << cnt << " " << current.norm() << "\n";
        if(current.norm() < tolerance) {
            break;
        }

        // BFGS公式
        if(cnt == 0) {
            H = Eigen::MatrixXd::Identity(N, N) / current.norm();
        } else {
            Eigen::VectorXd y = current - prev_val;
            Eigen::VectorXd s = estimation - prev_x;
            double sigma = 1.0 / s.dot(y);
            H = (Eigen::MatrixXd::Identity(N, N) - sigma * s * y.transpose()) * H * (Eigen::MatrixXd::Identity(N, N) - sigma * y * s.transpose()) + sigma * s * s.transpose();
        }
        Eigen::VectorXd delta = -H * current;
        double alpha=1.0;
        while(alpha > 1e-9) {
            if(target_func(estimation + alpha*delta).norm() < current.norm()) {
                break;
            } else {
                alpha *= 0.5;
            }
        }
        prev_x = estimation; prev_val = current;
        estimation += alpha * delta;
        cnt++;
    }

    if(cnt >= max_iter) cnt = -1;
    std::cout << "\n\n*********************************************\n\n";
    if(cnt < 0) {
        std::cout << "Quasi-Newton Method Did NOT Converge.\n";
    } else {
        std::cout << "Nonlinear equation solved!\n";
        std::cout << "Solver: BFGS method" << "\n";
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
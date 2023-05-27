#include <iostream>

#include <Eigen/Dense>

// solve non-linear equation by newton method.
// use analytic derivative if provided
// template parameter N is the dimension of input variable
// i.e. given f: R^N -> R^N, find x s.t. f(x) = 0
template<int N>
class NewtonMethod {
public:
    NewtonMethod(Eigen::VectorXd (*target_func)(Eigen::VectorXd), Eigen::MatrixXd (*jacob)(Eigen::VectorXd))
    : target_func(target_func), get_jacobian(jacob) {

    }
    Eigen::VectorXd solve(Eigen::VectorXd initial_point, double tolerance = 1e-9, int max_iteration = -1) {
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

private:
    Eigen::VectorXd (*target_func)(Eigen::VectorXd);
    Eigen::MatrixXd (*get_jacobian)(Eigen::VectorXd);
};

// example 3.7
Eigen::VectorXd example3_7(Eigen::VectorXd p) {
    Eigen::VectorXd ret = Eigen::VectorXd::Zero(p.rows());
    ret(0) = 2.0*p(0)*p(0)*p(0)+4.0*p(0)*p(0)+p(0)-2.0;
    return ret;
}

Eigen::MatrixXd example3_7_jacob(Eigen::VectorXd p) {
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(p.rows(), p.rows());
    ret(0, 0) = 6.0 * p(0) * p(0) + 8.0 * p(0) + 1.0;
    return ret;
}

// example 3.8
Eigen::VectorXd example3_8(Eigen::VectorXd p) {
    Eigen::VectorXd ret = Eigen::VectorXd::Zero(p.rows());
    ret(0) = p(1) - 1.0/p(0);
    ret(1) = p(1) - sqrt(p(0));
    return ret;
}

Eigen::MatrixXd example3_8_jacob(Eigen::VectorXd p) {
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(p.rows(), p.rows());
    ret(0, 0) = 1.0 / p(0) / p(0);
    ret(0, 1) = 1.0;
    ret(1, 0) = -0.5/sqrt(p(0));
    ret(1, 1) = 1.0;
    return ret;
}

// example 
int main() {
    NewtonMethod<1> nm1(example3_7, example3_7_jacob);
    Eigen::VectorXd sol = nm1.solve(1.5*Eigen::VectorXd::Ones(1));
    std::cout << sol << std::endl;
    sol = nm1.solve(-0.5*Eigen::VectorXd::Ones(1));
    std::cout << sol << std::endl;

    NewtonMethod<2> nm2(example3_8, example3_8_jacob);
    Eigen::VectorXd initial_point = Eigen::VectorXd::Zero(2);
    initial_point << 2.0, 3.0;
    sol = nm2.solve(initial_point);
    std::cout << sol << std::endl;

}
#pragma once

#include <iostream>
#include <Eigen/Dense>

// solve unconstrained optimization problem
class SearchDirection {
protected:
    double (*objective_func)(const Eigen::VectorXd&);
    Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&);

    Eigen::VectorXd optimal_point;
    double optimal_function_value;

    SearchDirection(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&))
    :objective_func(objective_func), gradient_func(gradient_func)  {

    }

public:
    Eigen::VectorXd get_optimal_point() {return optimal_point;};
    double get_optimal_value() {return optimal_function_value;};
};

template<class LineSearchAlgorithm>
class GradientDescent: public SearchDirection {
private:

public:
    GradientDescent(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&))
    : SearchDirection(objective_func, gradient_func) {

    }
    void solve(Eigen::VectorXd x, double tolerance=1e-9) {
        Eigen::VectorXd grad = gradient_func(x);
        double alpha = 1.0;
        int cnt = 0;
        while(grad.maxCoeff() > tolerance || -grad.minCoeff() > tolerance) {
            if(cnt > 0) grad = gradient_func(x);
            double grad_norm = grad.norm();
            Eigen::VectorXd search_direction = -grad / grad_norm;
            if(cnt > 0) {
                alpha /= grad_norm; 
            }
            LineSearchAlgorithm ls(objective_func, gradient_func, objective_func(x), -grad_norm, x, search_direction, alpha);
            x += ls.find_step() * search_direction;
            alpha *= grad_norm;
            ++cnt;
        }
        optimal_point = x;
        optimal_function_value = objective_func(x);
    };
};

template<class LineSearchAlgorithm>
class ConjugateGradient: public SearchDirection {
private:

public:
    ConjugateGradient(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&))
    : SearchDirection(objective_func, gradient_func) {

    }
    void solve(Eigen::VectorXd x, double tolerance=1e-9) {
        const int N = x.rows();
        Eigen::VectorXd grad, search_direction;
        search_direction = Eigen::VectorXd::Zero(N);
        grad = Eigen::VectorXd::Ones(N);
        int cnt = 0;
        bool reset = false;

        double grad_squared_norm = 1.0, prev_grad_squared_norm = 1.0;
        double beta = 0.0;
        while(grad.maxCoeff() > tolerance || -grad.minCoeff() > tolerance) {
            grad = gradient_func(x);

            prev_grad_squared_norm = grad_squared_norm;
            grad_squared_norm = grad.squaredNorm();

            if(reset) {
                beta = 0;
                reset = false;
            } else {
                beta = grad_squared_norm / prev_grad_squared_norm;
            }

            if(beta > 0.10) {
                reset = true;
            }

            search_direction = - grad / sqrt(grad_squared_norm) + beta * search_direction;
            double grad_phi0 = grad.dot(search_direction);

            LineSearchAlgorithm ls(objective_func, gradient_func, objective_func(x), grad_phi0, x, search_direction);

            x += ls.find_step() * search_direction;
            ++cnt;
        }
        optimal_point = x;
        optimal_function_value = objective_func(x);
    };
};

template<class LineSearchAlgorithm, class LinearSolver = Eigen::FullPivLU<Eigen::MatrixXd>>
class NewtonsMethod: public SearchDirection {
private:
    Eigen::MatrixXd (*hessian_func)(const Eigen::VectorXd&);
public:
    NewtonsMethod(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&), Eigen::MatrixXd (*hessian_func)(const Eigen::VectorXd&))
    : SearchDirection(objective_func, gradient_func), hessian_func(hessian_func) {

    }
    void solve(Eigen::VectorXd x, double tolerance=1e-9) {
        const int N = x.rows();
        Eigen::VectorXd grad = Eigen::VectorXd::Ones(N);
        Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(N, N);
        double alpha = 1.0;
        int cnt = 0;
        while(grad.maxCoeff() > tolerance || -grad.minCoeff() > tolerance) {
            grad = gradient_func(x);
            hessian = hessian_func(x);
            LinearSolver sol(hessian);
            Eigen::VectorXd search_direction = -sol.solve(grad);
            double grad_phi0 = grad.dot(search_direction);
            LineSearchAlgorithm ls(objective_func, gradient_func, objective_func(x), grad_phi0, x, search_direction);
            x += ls.find_step() * search_direction;
        }
        optimal_point = x;
        optimal_function_value = objective_func(x);
    };
};

template<class LineSearchAlgorithm>
class BFGS: public SearchDirection {
private:
    Eigen::MatrixXd H;

public:
    BFGS(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&))
    : SearchDirection(objective_func, gradient_func) {

    }

    void solve(Eigen::VectorXd x, double tolerance=1e-9) {
        const int N = x.rows();
        bool reset = false;
        int cnt = 0;
        Eigen::VectorXd s, y;
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(N);
        while(1) {
            y = grad;
            grad = gradient_func(x);
            y = grad - y;
            if(grad.maxCoeff() < tolerance && -grad.minCoeff() < tolerance) break; // converged.
            if(cnt == 0 || reset) {
                H = Eigen::MatrixXd::Identity(N, N) * grad.norm();
            } else {
                double sigma = 1.0 / s.dot(y);
                H = (Eigen::MatrixXd::Identity(N, N) - sigma * s * y.transpose()) * H * (Eigen::MatrixXd::Identity(N, N) - sigma * y * s.transpose()) + sigma * s * s.transpose();
            }
            Eigen::VectorXd search_direction = -H * grad;
            double grad_phi0 = grad.dot(search_direction);
            LineSearchAlgorithm ls(objective_func, gradient_func, objective_func(x), grad_phi0, x, search_direction);
            s = ls.find_step() * search_direction;
            x += s;
            ++cnt;
        }
        optimal_point = x;
        optimal_function_value = objective_func(x);
    }
};
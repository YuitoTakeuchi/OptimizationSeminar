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

    Eigen::VectorXd get_optimal_point() {return optimal_point;};
    double get_optimal_value() {return optimal_function_value;};
};
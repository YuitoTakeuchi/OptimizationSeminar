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
    
    const int max_iter;
    int cnt;
    std::vector<Eigen::VectorXd> point_history;
    std::vector<double> obj_history;
    bool store_points;

    SearchDirection(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&), int max_iter = 1000000)
    :objective_func(objective_func), gradient_func(gradient_func), max_iter(max_iter)  {
        store_points = false;
        cnt = 0;
    }

public:
    Eigen::VectorXd get_optimal_point() {return optimal_point;};
    double get_optimal_value() {return optimal_function_value;};
    std::vector<Eigen::VectorXd>& get_point_history() {return point_history;}
    std::vector<double>& get_obj_hisotry() {return obj_history;}

    void set_store_points(bool target) {
        store_points = target;
        if(target) {
            point_history.resize(max_iter);
            obj_history.resize(max_iter);
        }
    }

    int get_iter() {return cnt;}
};

template<class LineSearchAlgorithm>
class GradientDescent: public SearchDirection {
private:

public:
    GradientDescent(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&), int max_iter = 1000000)
    : SearchDirection(objective_func, gradient_func, max_iter) {

    }
    void solve(Eigen::VectorXd x, double tolerance=1e-9) {
        Eigen::VectorXd grad = gradient_func(x);
        double alpha = 1.0;
        while(grad.maxCoeff() > tolerance || -grad.minCoeff() > tolerance && cnt < max_iter) {
            double obj_val = objective_func(x);
            if(store_points) {
                point_history[cnt] = x;
                obj_history[cnt] = obj_val;
            }
            if(cnt > 0) grad = gradient_func(x);
            double grad_norm = grad.norm();
            Eigen::VectorXd search_direction = -grad / grad_norm;
            if(cnt > 0) {
                alpha /= grad_norm; 
            }
            LineSearchAlgorithm ls(objective_func, gradient_func, obj_val, -grad_norm, x, search_direction, alpha);
            x += ls.find_step() * search_direction;
            alpha *= grad_norm;
            ++cnt;
        }
        if(cnt == max_iter) cnt = -1;
        optimal_point = x;
        optimal_function_value = objective_func(x);
    };
};

template<class LineSearchAlgorithm>
class ConjugateGradient: public SearchDirection {
private:

public:
    ConjugateGradient(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&), int max_iter = 1000000)
    : SearchDirection(objective_func, gradient_func, max_iter) {

    }
    void solve(Eigen::VectorXd x, double tolerance=1e-9) {
        const int N = x.rows();
        Eigen::VectorXd grad, search_direction;
        search_direction = Eigen::VectorXd::Zero(N);
        grad = Eigen::VectorXd::Ones(N);
        bool reset = false;

        double grad_squared_norm = 1.0, prev_grad_squared_norm = 1.0;
        double beta = 0.0;
        while(grad.maxCoeff() > tolerance || -grad.minCoeff() > tolerance && cnt < max_iter) {
            double obj_val = objective_func(x);
            if(store_points) {
                point_history[cnt] = x;
                obj_history[cnt] = obj_val;
            }
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

            LineSearchAlgorithm ls(objective_func, gradient_func, obj_val, grad_phi0, x, search_direction);

            x += ls.find_step() * search_direction;
            ++cnt;
        }
        if(cnt == max_iter) cnt = -1;
        optimal_point = x;
        optimal_function_value = objective_func(x);
    };
};

template<class LineSearchAlgorithm, class LinearSolver = Eigen::FullPivLU<Eigen::MatrixXd>>
class NewtonsMethod: public SearchDirection {
private:
    Eigen::MatrixXd (*hessian_func)(const Eigen::VectorXd&);
public:
    NewtonsMethod(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&), Eigen::MatrixXd (*hessian_func)(const Eigen::VectorXd&), int max_iter = 1000000)
    : SearchDirection(objective_func, gradient_func, max_iter), hessian_func(hessian_func) {

    }
    void solve(Eigen::VectorXd x, double tolerance=1e-9) {
        const int N = x.rows();
        Eigen::VectorXd grad = Eigen::VectorXd::Ones(N);
        Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(N, N);
        double alpha = 1.0;
        while(grad.maxCoeff() > tolerance || -grad.minCoeff() > tolerance && cnt < max_iter) {
            double obj_val = objective_func(x);
            if(store_points) {
                point_history[cnt] = x;
                obj_history[cnt] = obj_val;
            }
            grad = gradient_func(x);
            hessian = hessian_func(x);
            LinearSolver sol(hessian);
            Eigen::VectorXd search_direction = -sol.solve(grad);
            double grad_phi0 = grad.dot(search_direction);
            LineSearchAlgorithm ls(objective_func, gradient_func, obj_val, grad_phi0, x, search_direction);
            x += ls.find_step() * search_direction;
            ++cnt;
        }
        if(cnt == max_iter) cnt = -1;
        optimal_point = x;
        optimal_function_value = objective_func(x);
    };
};
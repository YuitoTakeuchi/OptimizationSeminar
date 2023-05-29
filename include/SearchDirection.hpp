#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>

// solve unconstrained optimization problem
class SearchDirection {
public:
    void set_store_points(bool target) {
        store_points = target;
        if(target) {
            point_history.reserve(max_iter);
            obj_history.reserve(max_iter);
        }
    }

    void output_to_file(std::ofstream& ofs) {
        int N = point_history.size();
        if(N == 0) {
            std::cerr << "no log are stored in the optimizer\n";
            return;
        }
        int Nx = point_history[0].rows();
        ofs << std::setprecision(15);
        for(int i = 0; i < N; ++i) {
            for(int j = 0; j < Nx; ++j) {
                ofs << point_history[i](j) << " ";
            }
            ofs << obj_history[i] << "\n";
        }
    }

    void output_to_file(const std::string& path) {
        std::ofstream ofs(path);
        output_to_file(ofs);
        ofs.close();
    }


    int get_iter() {return cnt;}
    double get_optimal_value() {return optimal_function_value;};
    Eigen::VectorXd get_optimal_point() {return optimal_point;};
    std::vector<Eigen::VectorXd>& get_point_history() {return point_history;}
    std::vector<double>& get_obj_hisotry() {return obj_history;}

protected:
    double (*objective_func)(const Eigen::VectorXd&);
    Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&);

    std::string optimizer;
    std::string line_search;

    Eigen::VectorXd optimal_point;
    double optimal_function_value;
    
    const int max_iter;
    const int Nx;
    int cnt;
    std::vector<Eigen::VectorXd> point_history;
    std::vector<double> obj_history;
    bool store_points;

    SearchDirection(int Nx, double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&), int max_iter = 1000000)
    :Nx(Nx), objective_func(objective_func), gradient_func(gradient_func), max_iter(max_iter)  {
        store_points = true;
        cnt = 0;
    }

    void output_result() {
        std::cout << "\n\n*********************************************\n\n";
        if(cnt < 0) {
            std::cout << "Optimization Process Did NOT Converge.\n";
        } else {
            std::cout << "Optimize Success!\n";
            std::cout << "Solver: " << optimizer << "\n";
            std::cout << "Iteration Count: " << cnt << "\n";
            std::cout << "Minimum value: " << optimal_function_value << " ";
            std::cout << "at (";
            for(int i = 0; i < optimal_point.rows(); ++i) {
                std::cout << optimal_point(i) << (i == optimal_point.rows()-1 ? "" : ", ");
            }
            std::cout << ")\n";
        }
        std::cout << "\n\n*********************************************\n\n";
    }

};

template<class LineSearchAlgorithm>
class GradientDescent: public SearchDirection {
private:

public:
    GradientDescent(int Nx, double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&), int max_iter = 1000000)
    : SearchDirection(Nx, objective_func, gradient_func, max_iter) {
        optimizer = "Gradient Descent";
    }
    void solve(Eigen::VectorXd x, double tolerance=1e-9) {
        Eigen::VectorXd grad = gradient_func(x);
        double alpha = 1.0;
        while(grad.maxCoeff() > tolerance || -grad.minCoeff() > tolerance && cnt < max_iter) {
            double obj_val = objective_func(x);
            if(store_points) {
                point_history.push_back(x);
                obj_history.push_back(obj_val);
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
        if(cnt >= max_iter) cnt = -1;
        optimal_point = x;
        optimal_function_value = objective_func(x);
        output_result();
    };
};

template<class LineSearchAlgorithm>
class ConjugateGradient: public SearchDirection {
private:

public:
    ConjugateGradient(int Nx, double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&), int max_iter = 1000000)
    : SearchDirection(Nx, objective_func, gradient_func, max_iter) {
        optimizer = "Conjugate Gradient";
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
                point_history.push_back(x);
                obj_history.push_back(obj_val);
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
        if(cnt >= max_iter) cnt = -1;
        optimal_point = x;
        optimal_function_value = objective_func(x);
        output_result();
    };
};

template<class LineSearchAlgorithm, class LinearSolver = Eigen::FullPivLU<Eigen::MatrixXd>>
class NewtonMethod: public SearchDirection {
private:
    Eigen::MatrixXd (*hessian_func)(const Eigen::VectorXd&);
public:
    NewtonMethod(int Nx, double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&), Eigen::MatrixXd (*hessian_func)(const Eigen::VectorXd&), int max_iter = 1000000)
    : SearchDirection(Nx, objective_func, gradient_func, max_iter), hessian_func(hessian_func) {
        optimizer = "Newton's Method";

    }
    void solve(Eigen::VectorXd x, double tolerance=1e-9) {
        const int N = x.rows();
        Eigen::VectorXd grad = Eigen::VectorXd::Ones(N);
        Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(N, N);
        double alpha = 1.0;
        while(grad.maxCoeff() > tolerance || -grad.minCoeff() > tolerance && cnt < max_iter) {
            double obj_val = objective_func(x);
            if(store_points) {
                point_history.push_back(x);
                obj_history.push_back(obj_val);
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
        if(cnt >= max_iter) cnt = -1;
        optimal_point = x;
        optimal_function_value = objective_func(x);
        output_result();
    };
};

template<class LineSearchAlgorithm>
class BFGS: public SearchDirection {
private:
    Eigen::MatrixXd H;

public:
    BFGS(int Nx, double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&), int max_iter=1000000)
    : SearchDirection(Nx, objective_func, gradient_func, max_iter) {
        optimizer = "BFGS";
    }

    void solve(Eigen::VectorXd x, double tolerance=1e-9) {
        const int N = x.rows();
        bool reset = false;
        Eigen::VectorXd s, y;
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(N);
        while(1) {
            double obj_val = objective_func(x);
            if(store_points) {
                point_history.push_back(x);
                obj_history.push_back(obj_val);
            }
            y = grad;
            grad = gradient_func(x);
            y = grad - y;
            if((grad.maxCoeff() < tolerance && -grad.minCoeff() < tolerance) || cnt>max_iter) break; // converged.
            if(cnt == 0 || reset) {
                H = Eigen::MatrixXd::Identity(N, N) * grad.norm();
            } else {
                double sigma = 1.0 / s.dot(y);
                H = (Eigen::MatrixXd::Identity(N, N) - sigma * s * y.transpose()) * H * (Eigen::MatrixXd::Identity(N, N) - sigma * y * s.transpose()) + sigma * s * s.transpose();
            }
            Eigen::VectorXd search_direction = -H * grad;
            double grad_phi0 = grad.dot(search_direction);
            LineSearchAlgorithm ls(objective_func, gradient_func, obj_val, grad_phi0, x, search_direction);
            s = ls.find_step() * search_direction;
            x += s;
            ++cnt;
        }
        if(cnt >= max_iter) cnt = -1;
        optimal_point = x;
        optimal_function_value = objective_func(x);
        output_result();
    }
};
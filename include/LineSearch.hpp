#pragma once
#include <Eigen/Core>
#include <iostream>
#include <string>

// line searchによってステップ幅を決める
// given: f: R^N -> R
//       grad f: R^N -> R^N
class LineSearch {
protected:
    double (*objective_func)(const Eigen::VectorXd&);
    Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&);

    // phi0 = f(x_start), g_phi0 = p * grad f(x_start), where p is search direction
    const double phi0, g_phi0;

    const Eigen::VectorXd x_start;
    const Eigen::VectorXd direction;

    const double alpha_init;

    LineSearch(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&),
        double current, double grad, Eigen::VectorXd x_start, Eigen::VectorXd direction, double alpha_init=1.0);
};

// Armijo条件を満たすようにステップ幅を求める
class Armijo: public LineSearch {
public:
    Armijo(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&),
        double current, double grad, Eigen::VectorXd x_start, Eigen::VectorXd direction, double alpha_init=1.0);
    double find_step(double mu=1e-4, double rho=0.5, double alpha_min=1e-9);
};

// wolf条件を満たすようにステップ幅を求める
class Wolf: public LineSearch {
private:
    double find_alpha_p(double alpha_low, double alpha_high, double phi_low, double phi_high, int order=2);
    double pinpoint(double alpha_low, double alpha_high, double phi_low, double phi_high, double mu1, double mu2);
public:
    Wolf(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&),
        double current, double grad, Eigen::VectorXd x_start, Eigen::VectorXd direction, double alpha_init=1.0);
    double find_step(double mu1=1e-4, double mu2=0.9, double sigma=2.0);
};
#pragma once
#include <Eigen/Core>
#include <iostream>
#include <string>

// line searchによってステップ幅を決める
// given: f: R^N -> R
//       grad f: R^N -> R^N
class LineSearch {
private:
    double (*objective_func)(const Eigen::VectorXd&);
    Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&);

    // phi0 = f(x_start), g_phi0 = p * grad f(x_start), where p is search direction
    const double phi0, g_phi0;

    const Eigen::VectorXd x_start;
    const Eigen::VectorXd direction;

    const double alpha_init;

    double find_alpha_p(double alpha_low, double alpha_high, double phi_low, double phi_high, int order=2);
    double pinpoint(double alpha_low, double alpha_high, double phi_low, double phi_high, double mu1, double mu2);

public:
    LineSearch(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&),
        double current, double grad, Eigen::VectorXd x_start, Eigen::VectorXd direction, double alpha_init=1.0);
    double armijo(double mu=1e-4, double rho=0.5);
    double wolf(double mu1=1e-4, double mu2=0.9, double sigma=2.0);
    double find_step(std::string method = "armijo");
};
#include "LineSearch.hpp"

LineSearch::LineSearch(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&),
    double current, double grad, Eigen::VectorXd x_start, Eigen::VectorXd direction, double alpha_init)
: objective_func(objective_func), gradient_func(gradient_func), phi0(current), g_phi0(grad),
    x_start(x_start), direction(direction), alpha_init(alpha_init) {}

Armijo::Armijo(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&),
    double current, double grad, Eigen::VectorXd x_start, Eigen::VectorXd direction, double alpha_init)
: LineSearch(objective_func, gradient_func, current, grad, x_start, direction, alpha_init) {

}

double Armijo::find_step(double mu, double rho) {
    double alpha = alpha_init;
    while(objective_func(x_start + alpha * direction) > objective_func(x_start) + mu * alpha * g_phi0) {
        alpha = rho * alpha;
    }
    return alpha;
};

Wolf::Wolf(double (*objective_func)(const Eigen::VectorXd&), Eigen::VectorXd (*gradient_func)(const Eigen::VectorXd&),
    double current, double grad, Eigen::VectorXd x_start, Eigen::VectorXd direction, double alpha_init)
: LineSearch(objective_func, gradient_func, current, grad, x_start, direction, alpha_init) {

}

double Wolf::find_alpha_p(double alpha_low, double alpha_high, double phi_low, double phi_high, int order) {
    double g_phi_low  = gradient_func(x_start + alpha_low  * direction).dot(direction);
    double g_phi_high = gradient_func(x_start + alpha_high * direction).dot(direction);

    double beta1, beta2;
    double alpha; // return value
    switch(order) {
    case 2:
        alpha = 2.0 * alpha_low * (phi_high - phi_low) - g_phi_low * (alpha_high*alpha_high - alpha_low*alpha_low);
        alpha /= 2.0 * (phi_high - phi_low - g_phi_low * (alpha_high - alpha_low));
        break;
    case 3:
        beta1 = g_phi_low + g_phi_high - 3.0 * (phi_high - phi_low) / (alpha_high - alpha_low);
        beta2 = ((alpha_high > alpha_low) ? 1.0 : -1.0) * sqrt(beta1*beta1 - g_phi_low * g_phi_high);
        alpha = alpha_high - (alpha_high - alpha_low) * (g_phi_high + beta2 - beta1) / (g_phi_high - g_phi_low + 2.0*beta2); 
        break;
    default:
        std::cerr << "The order of interpolation must be 2 or 3. " << __LINE__ << "\n";
    }
    return alpha;
};

double Wolf::pinpoint(double alpha_low, double alpha_high, double phi_low, double phi_high, double mu1, double mu2) {
    int k = 0;
    double alpha; // return value
    while(1) {
        double alpha_p = find_alpha_p(alpha_low, alpha_high, phi_low, phi_high);
        double phi_p = objective_func(x_start + alpha_p * direction);
        if(phi_p > phi0 + mu1*alpha_p*g_phi0 || phi_p > phi_low) {
            alpha_high = alpha_p;
        } else {
            double g_phi_p = gradient_func(x_start + alpha_p * direction).dot(direction);
            if(std::abs(g_phi_p) <= -mu2 * g_phi0) {
                alpha = alpha_p;
                break;
            } else if(g_phi_p * (alpha_high - alpha_low) >= 0) {
                alpha_high = alpha_low;
            }
            alpha_low = alpha_p;
        }
        ++k;
    }

    return alpha;
};

double Wolf::find_step(double mu1, double mu2, double sigma) {
    double alpha1 = 0.0, alpha2 = alpha_init;
    double phi1 = phi0, g_phi1 = g_phi0, phi2;

    int cnt = 0;
    double alpha; // return value
    while(true) {
        phi1 = objective_func(x_start + alpha1 * direction);
        phi2 = objective_func(x_start + alpha2 * direction);
        if(phi2>phi0+mu1*alpha2*g_phi0 || (cnt > 0 && phi2 > phi1)) {
            alpha = pinpoint(alpha1, alpha2, phi1, phi2, mu1, mu2);
            break;
        }
        double g_phi2 = gradient_func(x_start + alpha2 * direction).dot(direction);
        if(std::abs(g_phi2) <= -mu2 * g_phi0) {
            alpha = alpha2;
            break;
        } else if(g_phi2 >= 0) {
            alpha = pinpoint(alpha2, alpha1, phi2, phi1, mu1, mu2);
            break;
        } else {
            alpha1 = alpha2;
            alpha2 = sigma * alpha2;
        }
        ++cnt;
    }

    return alpha;
};
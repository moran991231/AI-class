
#include <math.h>
#include "functions.h"
double sigmoid(double x) { 
    return 1.0/(1.0 + exp(-x));
}

double diff_sigmoid(double x) {
    double sig_x = sigmoid(x);
    return sig_x * (1.0 - sig_x);
}

double tanh_(double x) {
    return 2.0 / (1 + exp(-2.0 * x)) - 1.0;
}
double diff_tanh(double x) {
    double tanh_x = tanh_(x);
    return 1 - tanh_x * tanh_x;
}
double ReLU(double x) {
    return x >= 0.0 ? x : 0.0;
}
double diff_ReLU(double x) {
    return x >= 0 ? 1.0 : 0.0;
}

// -----------------------------------
double my_activate_func(double x) { 
    return tanh_(x);
}
double my_diff_activate_func(double x) {
    return diff_tanh(x);
}

double my_loss_func(double target, double y) {
    return (y - target) * (y - target) / 2.0;
}
double my_diff_loss_func(double target, double y) {
    y =my_decision_func(y);
    return y - target;
}

double my_decision_func(double x) {
    return x >= 0.5 ? 1 : 0;
}


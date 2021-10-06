
#include <math.h>
#include "functions.h"
double sigmoid(double x) { 
    return 1/(1.0 + exp(-x));
}

double diff_sigmoid(double x) {
    double sig_x = sigmoid(x);
    return sig_x * (1 - sig_x);
}

double ReLU(double x) {
    return (0.1 * x > x) ? 0.1 * x : x;
}
double diff_ReLU(double x) {
    return (0.1 * x > x) ? 0.1 : 1;
}

// -----------------------------------
double my_activate_func(double x) { 
    return ReLU(x);
}
double my_diff_activate_func(double x) {
    return diff_ReLU(x);
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


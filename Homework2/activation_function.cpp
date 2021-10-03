
#include <math.h>
#include "activation_function.h"
double sigmoid(double x) { 
    return 1/(1.0 + exp(-x));
}

double diff_sigmoid(double x) {
    double sig_x = sigmoid(x);
    return sig_x * (1 - sig_x);
}

double my_activate_func(double x) {// range (-2,2)
    double sig_x = sigmoid(x);
    return 4 * sig_x - 2;
}
double my_diff_activate_func(double x) {
    return 4 * diff_sigmoid(x);
}

double my_activate_func2(double x) {
    double sig_x = 4*sigmoid(x)-2;
    if (sig_x >= 1) return 1;
    else if (sig_x <= -0.1) return -0.1;
    return sig_x;
}
double my_diff_activate_func2(double x) {

    double sig_x = 4 * sigmoid(x) - 2;
    if (sig_x >= 1|| sig_x<=-0.1) return 0;
    return 4 * diff_sigmoid(x);
}


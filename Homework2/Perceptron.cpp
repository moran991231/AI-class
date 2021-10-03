#include "Perceptron.h"
#include <stdio.h>
#include<string.h>
Perceptron::Perceptron()
{
}
Perceptron::Perceptron(int num) {
    // make input and weight array
    num_input = num;
    arr_size = num + 1;
    input = new double[arr_size];
    weights = new double[arr_size];
    grad = new double[arr_size];
    d_weights = new double[arr_size];

}

Perceptron::Perceptron(const Perceptron& copy_source)
{
    num_input = copy_source.num_input;
    arr_size = copy_source.arr_size;
    input = new double[arr_size];
    weights = new double[arr_size];
    grad = new double[arr_size];
    d_weights = new double[arr_size];
    learning_rate = copy_source.learning_rate;
    activation_func = copy_source.activation_func;
    diff_activation_func = copy_source.diff_activation_func;
}

Perceptron::~Perceptron() {
    // delete input and weight array
    delete[] input;
    delete[] weights;
    delete[] grad;
    printf("finalize\n");
}

void Perceptron::set_input(double in[])
{
    memcpy(input, in, sizeof(double) * arr_size);
}

void Perceptron::set_weight(double w[])
{
    memcpy(weights, w, sizeof(double) * arr_size);
}

void Perceptron::set_d_weight(double d[])
{
    memcpy(d_weights, d, sizeof(double) * arr_size);
}

double Perceptron::feed_forward() {
    // calculate output
    net = 0.0;
    for (int i = 0; i < arr_size; i++)
        net += (input[i] * weights[i]);
    output = activation_func(net);
    return output;
}

void Perceptron::calc_grad(double target)
{
    // compare output with target(answer)
    // calculate gradient dy/dwi (y is Err, wi is weights[i])
    loss_func(target);
    double dErr_dNet =  diff_loss_func(target) * diff_activation_func(net);
    for (int i = 0; i < arr_size; i++)
        grad[i] = dErr_dNet * input[i];
}

double Perceptron::loss_func(double target)
{
    // loss function is (answer - output)^2/2
    error = (target - output) * (target - output) / 2.0;
    return error;
}

double Perceptron::diff_loss_func(double target) // differential of loss function
{
    double temp_output = output >= 0 ? 1 : 0;
    return temp_output - target;
}

void Perceptron::calc_d_weights()
{
    // calculate delta weights[i]
    for (int i = 0; i < arr_size; i++)
        d_weights[i] = -learning_rate * grad[i];
}

void Perceptron::update_weights()
{
    // w_new = w_old + delta_w
    for (int i = 0; i < arr_size; i++)
        weights[i] += d_weights[i];
}

void Perceptron::print_weights()
{
    for (int i = 0; i < arr_size; i++)
        printf("%lf ", weights[i]);
}





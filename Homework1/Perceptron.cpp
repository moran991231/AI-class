#include "Perceptron.h"

#include<string.h>
Perceptron::Perceptron(int num) {
    // make input and weight array
    num_input = num;
    arr_size = num + 1;
    input = new double[arr_size];
    weights = new double[arr_size];
}

Perceptron::~Perceptron() {
    // delete input and weight array
    delete[] input;
    delete[] weights;
}

void Perceptron::set_input(double in[])
{
    memcpy(input, in, sizeof(double) * arr_size);
}

void Perceptron::set_weight(double w[])
{
    memcpy(weights, w, sizeof(double) * arr_size);
}

double Perceptron::feed_forward() {
    // calculate output
    double temp = 0.0;
    for (int i = 0; i < arr_size; i++)
        temp += (input[i] * weights[i]);
    return (output = (temp >= 0.0 ? 1.0 : 0.0));
}

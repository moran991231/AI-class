#pragma once
class Perceptron
{
public:
    int num_input = 0, arr_size = 0; // arr_size == num_input + 1
    double* input = nullptr, * weights = nullptr;
    double output = 0.0;

    Perceptron(int num); // make input and weight array
    ~Perceptron();       // delete input and weight array

    void set_input(double in[]); // copy input data
    void set_weight(double w[]); // copy weight data
    double feed_forward();       // calculate output
};





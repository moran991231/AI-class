#pragma once
class Perceptron
{
public:
    int num_input = 0, arr_size = 0; // arr_size == num_input + 1
    double* input = nullptr, * weights = nullptr;
    double* grad = nullptr, * d_weights = nullptr;
    double learning_rate = 1;

    // result of calculation
    double output = 0.0, net =0.0;
    double error = 0.0;
    
    double (*activation_func)(double net);
    double (*diff_activation_func) (double net);
    Perceptron();
    Perceptron(int num); // make input and weight array
    Perceptron(const Perceptron& copy_source); // copy constructor
    ~Perceptron();       // delete input and weight array
    
    void set_input(double in[]); // copy input data
    void set_weight(double w[]); // copy weight data
    void set_d_weight(double d[]);
    double feed_forward();       // calculate output
    void calc_grad(double target);
     
    double loss_func(double target);
    double diff_loss_func(double target);
    void calc_d_weights();
    void update_weights();
    void print_weights();
};





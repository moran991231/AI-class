#pragma once
#include <stdio.h>
#include <vector>
#include "Layer.h"

using namespace std;
class Network
{
public:
    int num_input, num_output, num_layer;
    vector<double>  error, dErr_dy;
    vector<Layer*> layers;

    Network();
    Network(int num_input, int num_output, int num_layer, int ...);
    ~Network();

    void init(double learning_rate, mt19937 mt);

    void forward(vector<double> input, vector<double> target);
    void backward(vector<double> input);

    vector<double>* get_output();
    void write_all_weight(std::string prefix);
};



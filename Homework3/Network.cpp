#define _CRT_SECURE_NO_WARNINGS
#include "Network.h"
#include <cstdarg>
#include "functions.h"

Network::Network()
{
}

Network::Network(int num_input, int num_output, int num_layer, int ...)
{
    this->num_input = num_input;
    this->num_output = num_output;
    this->num_layer = num_layer;
    error.resize(num_output + 1);
    dErr_dy.resize(num_output + 1);

    va_list ap;
    va_start(ap, num_layer);
    int n = va_arg(ap, int);
    layers.push_back(new Layer(num_input, n));
    for (int i = 1; i < num_layer; i++) {
        n = va_arg(ap, int);
        layers.push_back(new Layer(layers[i - 1]->num_node, n));
    }
}

Network::~Network()
{
    for (int i = 0; i < num_layer; i++)
        delete layers[i];
}

void Network::init(double learning_rate, mt19937 mt)
{
    Layer* l;
    for (int i = 0; i < num_layer; i++) {
        l = layers[i];
        l->set_weight_randomly(mt);
        l->learning_rate = learning_rate;
        l->activation_func = my_activate_func;
        l->diff_activation_func = my_diff_activate_func;
    }
    l = layers[num_layer - 1];
    l->loss_func = my_loss_func;
    l->diff_loss_func = my_diff_loss_func;
}

void Network::forward(vector<double> input, vector<double> target)
{
    layers[0]->feed_forward(input); // input layer
    for (int i = 1; i < num_layer; i++)
        layers[i]->feed_forward(layers[i - 1]->output);
    layers[num_layer - 1]->apply_loss_func(target, error, dErr_dy); // output layer
}

void Network::backward(vector<double> input) {

    if (num_layer == 1) {
        layers[num_layer - 1]->feed_backward(input, dErr_dy);
    }
    else {
        layers[num_layer - 1]->feed_backward(layers[num_layer - 2]->output, dErr_dy); // input layer
        for (int i = num_layer - 2; 0 < i; i--) {
            layers[i]->feed_backward(layers[i - 1]->output, layers[i + 1]->del); // mid layer
        }
        layers[0]->feed_backward(input, layers[1]->del); // output layer
    }
}

vector<double>* Network::get_output()
{
    return &(layers[num_layer - 1]->output);
}

void Network::write_all_weight(string prefix)
{
    char filename[100] = "";
    for (int i = 0; i < num_layer; i++) {
        sprintf(filename, "%s%2d-th_layer_w.txt", prefix.c_str(), i);
        layers[i]->write_weight(std::string(filename));
    }
}

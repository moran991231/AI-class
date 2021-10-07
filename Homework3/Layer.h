#include <vector>
#include <string>
#include <random>
#pragma once
using namespace std;
class Layer
{
public: 
    int num_input=0, num_node = 0; // input:0~num_input, node(output): 1~num_node
    double learning_rate = 0.1;
    vector<double> del, net, output;
    vector<vector <double>> weight;

    double (*activation_func)(double net);
    double (*diff_activation_func) (double net);

    double (*loss_func)(double target, double y); // objective function
    double (*diff_loss_func) (double target, double y);

    Layer();
    Layer(int num_input, int num_node);
    Layer(const Layer& copy_source); // copy constructor

    // util functions
    void set_weight_randomly(std::mt19937& mt);
    void print_weight();
    void read_weight(std::string filename);
    void write_weight(std::string filename);

    void feed_forward(vector<double> input);  // forward process for all nodes
    void feed_backward(vector<double> input, vector<double> dErr_dy); // backward process for all nodes

    void apply_loss_func(vector<double> target, vector<double>& error, vector<double>& dErr_dy);

private: // a node (a perceptron)
    void perceptron_forward(vector<double> input,  int node_i);
    void perceptron_backward(double dErr_dy, vector<double> input, int node_i);
};


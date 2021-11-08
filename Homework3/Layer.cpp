#include "Layer.h"
#include <algorithm>
#include <fstream>
#include <stdio.h> 

Layer::Layer()
{
    // no use
}

Layer::Layer(int num_input, int num_node)
{
    this->num_input = num_input;
    this->num_node = num_node;
    del.resize(num_input + 1);
    net.resize(num_node + 1, 1);
    output.resize(num_node + 1, 1);
    weight.resize(num_node + 1, vector<double>(num_input + 1, 1));

}

Layer::Layer(const Layer& copy_source)
{
    num_input = copy_source.num_input;
    num_node = copy_source.num_node;
    learning_rate = copy_source.learning_rate;
    del.resize(num_input + 1);
    net.resize(num_node + 1, 1);
    output.resize(num_node + 1, 1);
    weight.resize(num_node + 1, vector<double>(num_input + 1, 1));

    activation_func = copy_source.activation_func;
    diff_activation_func = copy_source.diff_activation_func;

    loss_func = copy_source.loss_func;
    diff_loss_func = copy_source.diff_loss_func;
}

void Layer::perceptron_forward(vector<double> input, int node_i)
{   // a node (perceptron) calculate sum of input*weight
    double temp = 0.0;
    for (int i = 0; i <= num_input; i++)
        temp += input[i] * weight[node_i][i];
    net[node_i] = temp; // net
    output[node_i] = activation_func(temp); // f(net)
}

void Layer::perceptron_backward(double dErr_dyi, vector<double> input, int node_i)
{   // a node (perceptron) calculate gradient and update new weight
    double dErr_dNet = dErr_dyi * diff_activation_func(net[node_i]); // dErr/dNet
    for (int i = 0; i <= num_input; i++) {
        del[i] += dErr_dNet * weight[node_i][i]; // dErr/dxi = sum(dErr/dNet * w[...][i])        
        weight[node_i][i] += -learning_rate * dErr_dNet * input[i]; // update w
    }
}

void Layer::feed_forward(vector<double> input)
{   // forward for all nodes
    for (int i = 1; i <= num_node; i++)
        this->perceptron_forward(input, i);
}

void Layer::feed_backward(vector<double> input, vector<double> dErr_dy)
{    // backward for all nodes
    std::fill(del.begin(), del.end(), 0);
    for (int i = 1; i <= num_node; i++)
        perceptron_backward(dErr_dy[i], input, i);
}

void Layer::apply_loss_func(vector<double> target, vector<double>& error, vector<double>& dErr_dy)
{   // at the end  layer (output layer)
    for (int i = 1; i <= num_node; i++) {
        error[i] = loss_func(target[i - 1], output[i]);
        dErr_dy[i] = diff_loss_func(target[i - 1], output[i]);
    }
}

void Layer::set_weight_randomly(std::mt19937& mt)
{
    for (int i = 1; i <= num_node; i++) {
        for (int j = 0; j <= num_input; j++)
            weight[i][j] = (double)((int)(mt() % 100) - 50) / 50.0;
    }
}

void Layer::print_weight()
{
    for (int i = 1; i <= num_node; i++) {
        for (int j = 0; j <= num_input; j++)
            printf("%lf ", weight[i][j]);
        printf("\n");
    }
}

void Layer::read_weight(std::string filename)
{
    ifstream fin(filename.c_str());
    if (!fin.is_open()) return;
    for (int i = 1; i <= num_node; i++) {
        for (int j = 0; j <= num_input; j++)
            fin >> weight[i][j];
    }
    fin.close();
}

void Layer::write_weight(std::string filename)
{
    ofstream fout(filename.c_str());
    if (!fout.is_open()) return;
    for (int i = 1; i <= num_node; i++) {
        for (int j = 0; j <= num_input; j++) {
            fout << weight[i][j] << ' ';
        }
        fout << endl;
    }
    fout.close();
}


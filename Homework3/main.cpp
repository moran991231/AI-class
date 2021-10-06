#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include<random>
#include<string>
#include <sstream>
#include "Network.h"
#include "functions.h"
#include <fstream>

#define ERR_PRINT 0
#define OUTPUT_PRINT 1
#define DECISION_OUTPUT_PRINT 1
#define ERR_SUM_PRINT 1
#define WEIGHT_PRINT 1
#define SUCCEEDED_PRINT 1

using namespace std;

void print_output(double* all_out, int n);
void print_decision_output(double* all_out, int n);
void print_err(double* all_err, int n);
void print_succeeded(double* all_out, int n);


// global variables
mt19937 mt_rand;
vector<vector<double>> train_set({
    vector<double>({ 1.0, 0.0,0.0  }),
    vector<double>({ 1.0, 1.0,0.0  }),
    vector<double>({ 1.0, 0.0,1.0  }),
    vector<double>({ 1.0, 1.0 ,1.0 }),
    vector<double>({ 1.0, 0.5,1.0  }),
    vector<double>({ 1.0, 1.0, 0.5 }),
    vector<double>({ 1.0, 0.0, 0.5 }),
    vector<double>({ 1.0, 0.5, 0.0 }),
    vector<double>({ 1.0, 0.5,0.5  })
    });
vector<vector<double>>answer(9, vector<double>(1, 0));
int num_iter, gate_type;
 
Network* my_network;

void set_answer(int type) {
    // 1: and, 2: or, 3: xor, 4: doughnut
    switch (type) {
    case 1:
        answer[3][0] = 1;
        break;
    case 2:
        answer[1][0] = answer[2][0] = answer[3][0] = 1;
        break;
    case 3:
        answer[1][0] = answer[2][0] = 1;
        break;
    case 4:
        answer[8][0] = 1;
        break;
    }
}

void init() {
    cout << "Choose test type (1:and, 2:or, 3:xor, 4:doughnut): ";
    cin >> gate_type;
    set_answer(gate_type);

    double lr;
    cout << "enter learning rate: ";
    cin >> lr;

    cout << "enter iteration: ";
    cin >> num_iter;

    unsigned int seed;
    cout << "set seed: ";
    cin >> seed;

    mt_rand.seed(seed);

    my_network = new Network(2, 1, 2, 2, 1);
    my_network->init(lr, mt_rand);
}

void learning(int iteration) {
    int train_i=0, n = gate_type<=3? 4:9;
    train_i = iteration % n;
    double all_output[9], all_err[9];
    double err_sum=0.0;
    vector<double>* pout;
    if (train_i == 0) {
    #if OUTPUT_PRINT||DECISION_OUTPUT_PRINT||ERR_PRINT||ERR_SUM_PRINT||SUCCEEDED_PRINT
        for (int i = 0; i < n; i++) {
            my_network->forward(train_set[i], answer[i]);
            pout = my_network->get_output();
            all_output[i] = pout->at(1);
            all_err[i] = my_network->error[1];
            err_sum += all_err[i];
        }
    #if OUTPUT_PRINT
        print_output(all_output, n);
    #endif
    #if DECISION_OUTPUT_PRINT
        print_decision_output(all_output, n);
    #endif
    #if ERR_PRINT
        print_err(all_err, n);
    #endif
    #if ERR_SUM_PRINT
        printf(" [err sum: %+6lf] ", err_sum);
    #endif
    
    #if SUCCEEDED_PRINT
        print_succeeded(all_output, n);
    #endif
    #endif
    printf("\n");
    }
    my_network->forward(train_set[train_i], answer[train_i]);
    my_network->backward(train_set[train_i]);
}

int main()
{
    init();
    //my_network->layers[0]->read_weight(" 0-th_layer_w.txt");
    //my_network->layers[1]->read_weight(" 1-th_layer_w.txt");
    int n = gate_type <= 3 ? 4 : 9;
    for (int i = 0; i < num_iter*n; i++) {
        learning(i);
    }
    //my_network->write_all_weight();

    delete my_network;
}

void print_output(double* all_out, int n) {
    printf(" [out:");
    for (int i = 0; i < n; i++)
        printf(" %+6lf", all_out[i]);
    printf("] ");
}
void print_decision_output(double* all_out, int n) {
    printf(" [decision out:");
    for (int i = 0; i < n; i++)
        printf(" %d", (int)my_decision_func(all_out[i]));
    printf("] ");
}
void print_err(double* all_err, int n) {
    printf(" [err:");
    for (int i = 0; i < n; i++)
        printf(" %+6lf", all_err[i]);
    printf("] ");
}

void print_succeeded(double* all_out, int n) {
    int cnt = 0;
    for (int i = 0; i < n; i++)
        if (my_decision_func(all_out[i]) == my_decision_func(answer[i][0]))
            cnt++;
    printf(" [%s] ", cnt == n ? "!SUCCEEDED!" : "FAILED");
}
 
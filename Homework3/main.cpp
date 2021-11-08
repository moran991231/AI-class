#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include<random>
#include<string>
#include <sstream>
#include "Network.h"
#include "functions.h"
#include <fstream>

#define ERR_PRINT 1
#define OUTPUT_PRINT 1
#define DECISION_OUTPUT_PRINT 1
#define ERR_SUM_PRINT 1
#define SUCCEEDED_PRINT 1

#define WRITE_WEIGHT 0
#define WRITE_INTERVAL 200
#define WRITE_ERROR 0

using namespace std;

void print_output(double* all_out, int n);
void print_decision_output(double* all_out, int n);
void print_err(double* all_err, int n);
bool print_succeeded(double* all_out, int n);


// global variables
mt19937 mt_rand;
vector<vector<double>> train_set({
    vector<double>({ 1.0, 0.0, 0.0 }),
    vector<double>({ 1.0, 1.0, 0.0 }),
    vector<double>({ 1.0, 0.0, 1.0 }),
    vector<double>({ 1.0, 1.0, 1.0 }),
    vector<double>({ 1.0, 0.5, 1.0 }),
    vector<double>({ 1.0, 1.0, 0.5 }),
    vector<double>({ 1.0, 0.0, 0.5 }),
    vector<double>({ 1.0, 0.5, 0.0 }),
    vector<double>({ 1.0, 0.5, 0.5 })
    });
vector<vector<double>>answer(9, vector<double>(1, 0));
int num_iter, gate_type;

Network* my_network;
ofstream fout("error.txt");

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

    my_network = new Network(2, 1, 3, 2, 2, 1);
    my_network->init(lr, mt_rand);
}

void write_weight() {
    cout << "\n Do you want to save weight? (n: no,  else: yes):  ";
    char in;
    cin >> in;
    if (in == 'n' || in == 'N') return;
    my_network->write_all_weight("");
}

bool learning(int iteration) {
    int n = gate_type <= 3 ? 4 : 9;
    double all_output[9], all_err[9];
    double err_sum = 0.0;
    vector<double>* pout;
    bool ret = false;

    for (int i = 0; i < n; i++) {
        my_network->forward(train_set[i], answer[i]);
        my_network->backward(train_set[i]);
    }
    printf("epoch %-4d: ", iteration);
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

    ret = print_succeeded(all_output, n);
    printf("\n");

#if WRITE_WEIGHT
    if (iteration % WRITE_INTERVAL == 0) {
        string prefix = "./weight/";
        prefix += to_string(iteration / WRITE_INTERVAL);
        prefix += " ";
        my_network->write_all_weight(prefix);
    }
#endif
#if WRITE_ERROR
    fout << err_sum << endl;
#endif
    return ret;
}

int main()
{
    init(); // set gate_type, learning_rate, num_iter, rand_seed 
    for (int i = 0; i < num_iter; i++)
        if (learning(i))     // 1 epoch
            break;           // criterion: first success

    write_weight();
    fout.close();
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

bool print_succeeded(double* all_out, int n) {
    int cnt = 0;
    for (int i = 0; i < n; i++)
        if (my_decision_func(all_out[i]) == (int)answer[i][0])
            cnt++;
#if SUCCEEDED_PRINT
    printf(" [%s] ", cnt == n ? "!SUCCEEDED!" : "FAILED");
#endif
    return cnt == n;
}

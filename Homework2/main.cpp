#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include<random>
#include<string>
#include <sstream>
#include "Perceptron.h"
#include "activation_function.h"
#include <fstream>

#define ERR_PRINT 0
#define OUTPUT_PRINT 0
#define DECISION_OUTPUT_PRINT 1
#define ERR_SUM_PRINT 1
#define WEIGHT_PRINT 1

using namespace std;


// global variables
mt19937 mt_rand;
ofstream fout("output.txt");
double ans_and[4] = { 0,0,0,1 }; // the truth table of AND operation
double ans_or[4] = { 0,1,1,1 }; // the truth table of OR operation
double ans_xor[4] = { 0,1,1,0 }; // the truth table of XOR operation
double cases[4][3] = { {1,0,0},{1,1,0},{1,0,1},{1,1,1} };
double* answer;
int num_iter;

int dicision_func(double output);
void error_print(Perceptron* percept_arr[]);
void  output_print(Perceptron* percept_arr[]);
void decision_output_print(Perceptron* percept_arr[]);
void weight_print(Perceptron* p);
void get_random_weight(double*);
void init(Perceptron** percept_arr);
void learing(Perceptron* percept_arr[], int num);
int get_max_i(Perceptron* percept_arr[]);

int main()
{
    Perceptron* my_perceptrons[4];
    init(my_perceptrons);
    
    for (int i = 0; i < num_iter; i++) { // iterate
        printf("%3d:", i);
        learing(my_perceptrons,2);
    }
    fout.close();
    return 0;
}

void get_random_weight(double* weight) { // random value [-1,1)
    for (int i = 0; i < 3; i++) {
        weight[i] = ((int)(mt_rand() % 100) - 50) / 50.0;
        cout << weight[i] << ' ';
    }
    cout << endl;
}

void init(Perceptron** percept_arr) {

    // select type of gate, and get the number of iteration
    
    cout << "What kind of gate do you want to test? (1: and, 2: or, 3: xor): ";
    int gate_type;
    cin >> gate_type;
    cout << "How many times do you want to iterate?  ";
    cin >> num_iter;
    cout << "Please enter seed: ";
    
    unsigned int seed;
    cin >> seed;
    mt_rand = mt19937(seed);
    if (gate_type == 1)
        answer = ans_and;
    else if (gate_type == 2)
        answer = ans_or;
    else
        answer = ans_xor;

    // set perceptron properties
    percept_arr[0] = new Perceptron(2);
    percept_arr[0]->activation_func = my_activate_func;
    percept_arr[0]->diff_activation_func = my_diff_activate_func;
    percept_arr[0]->learning_rate = 100; // learning rate
    percept_arr[0]->set_input(cases[0]);

    // apply same properties to other perceptrons
    for (int i = 1; i < 4; i++) {
        percept_arr[i] = new Perceptron(*percept_arr[0]);
        percept_arr[i]->set_input(cases[i]);
    }

    // initially, weights have random values
    double weights[3] = {};
    get_random_weight(weights);
    for (int i = 0; i < 4; i++)
        percept_arr[i]->set_weight(weights);

}


void learing(Perceptron* percept_arr[], int num) {
    Perceptron* p = nullptr;
    double d_weights[3] = { 0 };
    double err_sum = 0;
    int arr_size = percept_arr[0]->arr_size;
    for (int i = 0; i < 4; i++) {
        p = percept_arr[i];
        p->feed_forward();
        p->calc_grad(answer[i]); // calculate gradient dy/dxi
        p->calc_d_weights();
        for (int j = 0; j < arr_size; j++) // sum the delta weights of each cases (00, 01, 10, 11)
            d_weights[j] += p->d_weights[j];
        err_sum += p->error;
    }

    for (int j = 0; j < arr_size; j++) // get average of delta weights
        d_weights[j] /= 4.0;

    // print option
#if OUTPUT_PRINT
    output_print(percept_arr);
#endif
#if DECISION_OUTPUT_PRINT
    decision_output_print(percept_arr);
#endif
#if ERR_PRINT
    error_print(percept_arr);
#endif
#if  ERR_SUM_PRINT
    printf("[total err: %lf]  ", err_sum);
#endif

    percept_arr[0]->set_d_weight(d_weights); // set delta_weights as average delta_weights
    num = 0;
    //num = get_max_i(percept_arr);
    percept_arr[num]->update_weights(); // set pack propagation

#if  WEIGHT_PRINT // print new weight
    weight_print(percept_arr[0]);
#endif

    // set weights 4 perceptons have same weights
    double* weights = percept_arr[num]->weights;
    for (int i = 0; i < 4; i++)
        percept_arr[i]->set_weight(weights);
    printf("\n");

    for (int i = 0; i < arr_size; i++) {
        fout<< weights[i]<< ' ';
    }
    fout << endl;
}


int dicision_func(double output) {
    return (output >= 0.0) ? 1 : 0;
}

void error_print(Perceptron* percept_arr[]) {
    printf("[err: ");
    for (int i = 0; i < 4; i++) {
        printf("%lf ", percept_arr[i]->error);
    }
    printf("]  ");
}
void  output_print(Perceptron* percept_arr[]) {
    printf("[output: ");
    for (int i = 0; i < 4; i++) {
        printf("%lf ", percept_arr[i]->output);
    }
    printf("]  ");
}
void decision_output_print(Perceptron* percept_arr[]) {
    printf("[decision: ");
    for (int i = 0; i < 4; i++) {
        printf("%d ", (percept_arr[i]->output >= 0) ? 1 : 0);
    }
    printf("]  ");
}
void weight_print(Perceptron* p) {
    printf("[weights: ");
    p->print_weights();
    printf("]  ");
}

int get_max_i(Perceptron* percept_arr[]) {
    double max = -100;
    int max_i = 0;
    for (int i = 0; i < 4; i++) {
        if (percept_arr[i]->error > max) {
            max = percept_arr[i]->error;
            max_i = i;
        }
    }
    return max_i;
}
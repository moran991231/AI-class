#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include<random>
#include<string>
#include <sstream>
#include "Perceptron.h"

using namespace std;

#define SEED 1234u // random seed
mt19937 mt_rand(SEED);
const int answer[4] = { 0,0,0,1 }; // the truth table of AND operation

bool get_new_weight(double*);
void get_random_weight(double*);
bool apply_new_weight(Perceptron[], double[]);

bool get_new_weight(double* weight) {
    // get command from user about what to do
    // quit: escape the while loop in the main()
    // random: set random weight
    // other: get weight manually
    cout << "Enter new weights (q:Quit r: random): ";
    char buf[100];
    cin.getline(buf, 99);
    if (buf[0] == 'q') return false;
    else if (buf[0] == 'r')
        get_random_weight(weight);
    else
        sscanf(buf, "%lf %lf %lf", weight, weight + 1, weight + 2);
    return true;
}

void get_random_weight(double* weight) {
    for (int i = 0; i < 3; i++) {
        weight[i] = (int)(mt_rand() % 100) - 50;
        cout << weight[i] << ' ';
    }
    cout << endl;
}

bool apply_new_weight(Perceptron percept_arr[], double weights[]) {
    // calculate new output and compare with the truth table
    int out;
    int cnt = 0;
    for (int i = 0; i < 4; i++) {
        percept_arr[i].set_weight(weights);
        out = (int)percept_arr[i].feed_forward();
        cout << out << ' ';
        if (answer[i] == out) cnt++;
    }
    if (cnt == 4) {
        cout << "SUCCEEDED" << endl;
        return true;
    }
    else {
        cout << "FAILED" << endl;
        return false;
    }
}

int main()
{
    Perceptron my_perceptrons[] = { Perceptron(2), Perceptron(2),  Perceptron(2),  Perceptron(2) };
    my_perceptrons[0].set_input(new double[] {1, 0, 0});
    my_perceptrons[1].set_input(new double[] {1, 0, 1});
    my_perceptrons[2].set_input(new double[] {1, 1, 0});
    my_perceptrons[3].set_input(new double[] {1, 1, 1});

    double weights[3] = {};
    int cnt = 1;
    while (true) {
        if (cnt > 10000) break;
        cout << cnt++ << " trial: "; // print trial count

        if (get_new_weight(weights) == false) break; // enter weights manually
        //get_random_weight(weights);                // get random weights automatically

        if (apply_new_weight(my_perceptrons, weights)) // test and check the result of new weights
            break;
    }

    return 0;
}

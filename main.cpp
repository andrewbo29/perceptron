#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

#ifndef PERCEPTRON
#define PERCEPTRON
#include "Perceptron.h"
#endif

int main()
{
    vector<vector<double>> data;
    vector<int> labels;

    enterData(data, labels);

    Perceptron perc = Perceptron(data);

    int iter_number = 5;

    perc.train(data, labels, iter_number);

    perc.prediction(data, labels);

    return 0;
}

void setMisclass(const vector<vector<double>> *data, const vector<int> *labels, Perceptron *perc,
                 vector<vector<double>> &misClass, vector<int> &misClassLabels,
                 vector<vector<double>> &trueClass, vector<int> &trueClassLabels) {
    misClass.clear();
    misClassLabels.clear();
    trueClass.clear();
    trueClassLabels.clear();

    vector<double> *percWeights = perc->get_weights();

    for (decltype(data->size()) i = 0; i < data->size(); ++i) {
        vector<double> data_elem = (*data)[i];
        double sum = 0;
        for (decltype(data_elem.size()) j = 0; j < data_elem.size(); ++j) {
            sum += (*percWeights)[j] * data_elem[j];
        }
        int pred = (sum > 0) ? 1 : -1;

        if (pred != (*labels)[i]) {
            misClass.push_back(data_elem);
            misClassLabels.push_back((*labels)[i]);
        }
        else {
            trueClass.push_back(data_elem);
            trueClassLabels.push_back((*labels)[i]);
        }
    }

    return;
}

void enterData (vector<vector<double>> &data, vector<int> &labels) {
    char c = 'y';

    while (c == 'y') {
        cout << "Enter 2-dim data" << endl;
        vector<double> data_point;
        double v1 = 0.0;
        double v2 = 0.0;
        cin >> v1 >> v2;
        int label = 0;
        cout << "Data point is written\n" << "Enter label (-1 or 1)" << endl;
        cin >> label;
        try {
            if (label != 1 && label != -1) {
                throw runtime_error("Label must be 1 or -1");
            }
            data_point.push_back(v1);
            data_point.push_back(v2);
            data_point.push_back(1.0);
            data.push_back(data_point);
            labels.push_back(label);
        }
        catch (runtime_error err) {
            cout << err.what() << endl;
        }
        cout << "Do you want to enter another data point (y/n)?" << endl;
        cin >> c;
        while (c != 'y' && c != 'n') {
            cin >> c;
        }
    }

    return;
}

Perceptron::Perceptron(const vector<vector<double>> &data) {
    for (decltype(data[0].size()) i = 0; i < data[0].size(); ++i) {
        double init_weight = static_cast<double>(rand()) / (RAND_MAX);
        weights.push_back(init_weight);
    }

    cout << "Weights" << endl;
    for (auto w : weights) {
        cout << w << "\t";
    }
    cout << endl;
}

void Perceptron::train(vector<vector<double>> &data, vector<int> &labels, int iter_number) {

    vector<vector<double>> misClass;
    vector<int> misClassLabels;

    vector<vector<double>> trueClass;
    vector<int> trueClassLabels;

    setMisclass(&data, &labels, this, misClass, misClassLabels, trueClass, trueClassLabels);

    int done_iter = 0;
    while (!misClass.empty() && done_iter < iter_number) {
        cout << "Learning iteration number " << done_iter << endl;
        ++done_iter;

        int max_ind = misClass.size();
        int rand_ind = rand() % max_ind;
        vector<double> rand_elem = misClass[rand_ind];

        //cout << rand_elem.size() << "\t" << weights.size() << endl;

        for (decltype(weights.size()) k = 0; k < weights.size(); ++k) {
            weights[k] += misClassLabels[rand_ind] * rand_elem[k];
        }

        setMisclass(&data, &labels, this, misClass, misClassLabels, trueClass, trueClassLabels);

        cout << "Weights" << endl;
        for (auto w : weights) {
            cout << w << "\t";
        }
        cout << endl;
    }

    return;
}

vector<double> *Perceptron::get_weights() {
    return &weights;
}

void Perceptron::prediction(vector<vector<double>> &data, vector<int> &labels) {
    for (decltype(data.size()) i = 0; i < data.size(); ++i) {
        vector<double> data_elem = data[i];
        double sum = 0;
        for (decltype(data_elem.size()) j = 0; j < data_elem.size(); ++j) {
            sum += weights[j] * data_elem[j];
        }
        int pred = (sum > 0) ? 1 : -1;

        cout << "Data: ";
        for (auto data_point : data_elem) {
            cout << data_point << " ";
        }
        cout << "Label: " << labels[i] << " " << "Predict: " << pred << endl;
    }

    char c1;
    while (cin >> c1) {
        cout << c1 << endl;
    }

    return;
}

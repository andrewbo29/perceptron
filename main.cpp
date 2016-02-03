#include <iostream>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <fstream>

using namespace std;

#ifndef PERCEPTRON
#define PERCEPTRON
#include "Perceptron.h"
#endif

#include "imageProcessing.h"

const string Perceptron::name = "Perceptron";

int main()
{
//    vector<vector<double>> data;
//    vector<int> labels;
//
////    enterData(data, labels);
//    readData("/home/boyarov/Projects/cpp/perceptron/data.txt", data, labels);

    vector<int> labels;

    string dirName = "/home/boyarov/Projects/cpp/data/mnist_data_0/";
    vector<vector<double>> data = readImagesDir(dirName);
    labels.insert(labels.end(), data.size(), -1);

    dirName = "/home/boyarov/Projects/cpp/data/mnist_data_1/";
    vector<vector<double>> newData = readImagesDir(dirName);
    data.insert(data.end(), newData.begin(), newData.end());
    labels.insert(labels.end(), newData.size(), 1);

    Perceptron perc = Perceptron(data);

    cout << "\nStart " << Perceptron::getName() << " learning" << endl;

    int iter_number = 5;

    perc.train(data, labels, iter_number);

    vector<int> predict = perc.predict(data);

    cout << predict.size() << endl;

//    showPredictResults(data, labels, predict);

    return 0;
}

void setMisclass(const vector<vector<double>> *data, const vector<int> *labels, const Perceptron *perc,
                 vector<vector<double>> &misClass, vector<int> &misClassLabels,
                 vector<vector<double>> &trueClass, vector<int> &trueClassLabels) {
    misClass.clear();
    misClassLabels.clear();
    trueClass.clear();
    trueClassLabels.clear();

    for (decltype(data->size()) i = 0; i < data->size(); ++i) {
        vector<double> data_elem = (*data)[i];
        double sum = 0;
        for (decltype(data_elem.size()) j = 0; j < data_elem.size(); ++j) {
            sum += perc->weights[j] * data_elem[j];
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

void enterData(vector<vector<double>> &data, vector<int> &labels) {
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

    cout << "Initial weights:" << endl;
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
        cout << "\nLearning iteration number " << done_iter << endl;
        ++done_iter;

        unsigned long max_ind = misClass.size();
        unsigned long rand_ind = rand() % max_ind;
        vector<double> &rand_elem = misClass[rand_ind];

        for (decltype(weights.size()) k = 0; k < weights.size(); ++k) {
            weights[k] += misClassLabels[rand_ind] * rand_elem[k];
        }

        setMisclass(&data, &labels, this, misClass, misClassLabels, trueClass, trueClassLabels);

        cout << "Weights:" << endl;
        for (auto w : weights) {
            cout << w << "\t";
        }
        cout << endl;
    }

    return;
}

vector<int> Perceptron::predict(vector<vector<double>> &data) {
    vector<int> predicted;

    for (decltype(data.size()) i = 0; i < data.size(); ++i) {
        vector<double> &data_elem = data[i];
        double sum = 0;
        for (decltype(data_elem.size()) j = 0; j < data_elem.size(); ++j) {
            sum += weights[j] * data_elem[j];
        }
        int pred = (sum > 0) ? 1 : -1;

        predicted.push_back(pred);
    }

    return predicted;
}

void showPredictResults(vector<vector<double>> &data, vector<int> &labels, vector<int> &predicted) {
    cout << "\nPrediction results:" << endl;

    for (decltype(data.size()) i = 0; i < data.size(); ++i) {
        vector<double> &data_elem = data[i];
        cout << "Data point: ";
        for (decltype(data_elem.size()) j = 0; j < data_elem.size() - 1; ++j) {
            cout << data_elem[j] << " ";
        }
        cout << "Label: " << labels[i] << " " << "Predict: " << predicted[i] << endl;
    }

    cin.get();
    while (1) {
        if (cin.get() == '\n') {
            return;
        }
    }
}

void readData(string fileName, vector<vector<double>> &data, vector<int> &labels) {
    string line;
    int label;
    double dataPoint;

    ifstream input(fileName);

    while (getline(input, line)) {
        vector<double> dataElem;
        istringstream record(line);
        if (record >> label) {
            labels.push_back(label);
        } else {
            cerr << "Error in reading data" << endl;
            return;
        }

        while (record >> dataPoint) {
            dataElem.push_back(dataPoint);
        }
        dataElem.push_back(1.0);
        data.push_back(dataElem);
    }

    return;
}

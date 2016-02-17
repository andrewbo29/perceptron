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

//    enterData(data, labels);
//    readData("/home/boyarov/Projects/cpp/perceptron/data.txt", data, labels);

//    string posDirName = "/home/boyarov/Projects/cpp/data/mnist_data_0/";
//    string negDirName = "/home/boyarov/Projects/cpp/data/mnist_data_1/";

    cout << "Load train data" << endl;

    vector<vector<double>> dataTrain;
    vector<int> labelsTrain;

    string posDirNameTrain = "/media/datab/bases/mnist/train/0";
    string negDirNameTrain = "/media/datab/bases/mnist/train/1";

    readImagesData(posDirNameTrain, negDirNameTrain, dataTrain, labelsTrain);

    Perceptron perc = Perceptron(dataTrain);

    cout << "\nStart " << Perceptron::getName() << " learning" << endl;

    int iter_number = 1000;

    perc.train(dataTrain, labelsTrain, iter_number);

    dataTrain.clear();
    labelsTrain.clear();

    cout << "Load test data" << endl;

    vector<vector<double>> dataTest;
    vector<int> labelsTest;

    string posDirNameTest = "/media/datab/bases/mnist/test/0";
    string negDirNameTest = "/media/datab/bases/mnist/test/1";

    readImagesData(posDirNameTest, negDirNameTest, dataTest, labelsTest);

    vector<int> predict = perc.predict(dataTest);

    cout << "\nPrediction error: " << getPredictionError(labelsTest, predict) << endl;

    return 0;
}

double setMisclass(const vector<vector<double>> *data, const vector<int> *labels, const Perceptron *perc,
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

    return double(misClass.size()) / data->size();
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
}

void Perceptron::train(vector<vector<double>> &data, vector<int> &labels, int iter_number) {
    vector<vector<double>> misClass;
    vector<int> misClassLabels;

    vector<vector<double>> trueClass;
    vector<int> trueClassLabels;

    setMisclass(&data, &labels, this, misClass, misClassLabels, trueClass, trueClassLabels);

    int done_iter = 0;
    while (!misClass.empty() && done_iter < iter_number) {
        ++done_iter;

        unsigned long max_ind = misClass.size();
        unsigned long rand_ind = rand() % max_ind;
        vector<double> &rand_elem = misClass[rand_ind];

        for (decltype(weights.size()) k = 0; k < weights.size(); ++k) {
            weights[k] += misClassLabels[rand_ind] * rand_elem[k];
        }

        trainError = setMisclass(&data, &labels, this, misClass, misClassLabels, trueClass, trueClassLabels);

        if (done_iter % 50 == 0) {
            cout << "\nLearning iteration number " << done_iter << endl;
            cout << "Train error: " << trainError << endl;
        }
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

void readImagesData(string posDirName, string negDirName, vector<vector<double>> &data, vector<int> &labels) {
    vector<vector<double>> posData = readImagesDir(posDirName);
    data.insert(data.end(), posData.begin(), posData.end());
    labels.insert(labels.end(), posData.size(), 1);

    vector<vector<double>> negData = readImagesDir(negDirName);
    data.insert(data.end(), negData.begin(), negData.end());
    labels.insert(labels.end(), negData.size(), -1);
}

double getPredictionError(vector<int> &labels, vector<int> &predicted) {
    int misClassSum = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] != predicted[i]) {
            misClassSum += 1;
        }
    }

    return double(misClassSum) / labels.size();
}

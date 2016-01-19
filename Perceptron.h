#include <vector>
using namespace std;

class Perceptron
{
public:
    Perceptron() = default;
    Perceptron(const vector<vector<double>> &data);
    vector<double> *get_weights();
    void train(vector<vector<double>> &data, vector<int> &labels, int iter_number);
    void prediction(vector<vector<double>> &data, vector<int> &labels);
private:
    std::vector<double> weights;
};

void setMisclass(const vector<vector<double>> *data, const vector<int> *labels, Perceptron *perc,
                 vector<vector<double>> &misClass, vector<int> &misClassLabels,
                 vector<vector<double>> &trueClass, vector<int> &trueClassLabels);

void enterData (vector<vector<double>> &data, vector<int> &labels);

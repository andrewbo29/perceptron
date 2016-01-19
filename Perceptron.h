#include <vector>
using namespace std;

struct Perceptron
{
    std::vector<double> weights;
};

void set_misclass(const vector<vector<double>> *data, const vector<int> *labels, const Perceptron *perc,
                  vector<vector<double>> &misClass, vector<int> &misClassLabels,
                  vector<vector<double>> &trueClass, vector<int> &trueClassLabels);

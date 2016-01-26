#include <vector>
using namespace std;

class Perceptron
{
public:
    Perceptron() = default;
    Perceptron(const vector<vector<double>> &data);
    void train(vector<vector<double>> &data, vector<int> &labels, int iter_number);
    inline vector<int> predict(vector<vector<double>> &data);

    friend void setMisclass(const vector<vector<double>> *data, const vector<int> *labels, const Perceptron *perc,
                            vector<vector<double>> &misClass, vector<int> &misClassLabels,
                            vector<vector<double>> &trueClass, vector<int> &trueClassLabels);
    static string getName() { return name;}
private:
    const static string name;
    std::vector<double> weights;
};

void enterData(vector<vector<double>> &data, vector<int> &labels);

void showPredictResults(vector<vector<double>> &data, vector<int> &labels, vector<int> &predicted);

void readData(string fileNmae, vector<vector<double>> &data, vector<int> &labels);

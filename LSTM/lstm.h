#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}
double sigmoidDerivatativ(double x){
    return x * (1.0 - x);
}

class LSTM {
private:
    std::vector<int> layerSizes; // {input, hidden1, hidden2, ..., output}
    double learningRate;
    std::mt19937 generator;
    std::uniform_real_distribution<double> dist;

    struct LSTMLayer {
        std::vector<std::vector<double>> Wf, Uf;
        std::vector<std::vector<double>> Wi, Ui;
        std::vector<std::vector<double>> Wo, Uo;
        std::vector<std::vector<double>> Wc, Uc;
        std::vector<double> hiddenState;
        std::vector<double> cellState;
    };

    std::vector<LSTMLayer> layers;
    std::vector<std::vector<double>> outputWeights;
    std::vector<double> outputLayer;

    void initializeWeight(std::vector<std::vector<double>>& weight, int rows, int cols){
        weight.resize(rows, std::vector<double>(cols));
        for(int i=0; i<rows; i++)
            for(int j=0; j<cols; j++)
                weight[i][j] = dist(generator);
    }

public:
    LSTM(const std::vector<int>& sizes, double lr)
        : layerSizes(sizes), learningRate(lr),
          generator(std::random_device{}()), dist(-1.0, 1.0)
    {
        int numHiddenLayers = sizes.size() - 2;
        layers.resize(numHiddenLayers);

        for(int l=0; l<numHiddenLayers; l++){
            int inSize = (l==0 ? sizes[0] : sizes[l]);
            int hidSize = sizes[l+1];

            layers[l].hiddenState.resize(hidSize, 0.0);
            layers[l].cellState.resize(hidSize, 0.0);

            initializeWeight(layers[l].Wf, inSize, hidSize);
            initializeWeight(layers[l].Uf, hidSize, hidSize);

            initializeWeight(layers[l].Wi, inSize, hidSize);
            initializeWeight(layers[l].Ui, hidSize, hidSize);

            initializeWeight(layers[l].Wo, inSize, hidSize);
            initializeWeight(layers[l].Uo, hidSize, hidSize);

            initializeWeight(layers[l].Wc, inSize, hidSize);
            initializeWeight(layers[l].Uc, hidSize, hidSize);
        }

        int lastHiddenSize = sizes[sizes.size()-2];
        int outSize = sizes.back();
        outputLayer.resize(outSize, 0.0);
        initializeWeight(outputWeights, lastHiddenSize, outSize);
    }

    std::vector<double> forward(const std::vector<double>& input){
        std::vector<double> currentInput = input;

        for(size_t l=0; l<layers.size(); l++){
            int hidSize = layers[l].hiddenState.size();
            std::vector<double> f(hidSize), i(hidSize), o(hidSize), c_tilde(hidSize);
            std::vector<double> newHidden(hidSize), newCell(hidSize);

            for(int k=0; k<hidSize; k++){
                double f_sum=0, i_sum=0, o_sum=0, c_sum=0;

                for(size_t j=0; j<currentInput.size(); j++){
                    f_sum += currentInput[j]*layers[l].Wf[j][k];
                    i_sum += currentInput[j]*layers[l].Wi[j][k];
                    o_sum += currentInput[j]*layers[l].Wo[j][k];
                    c_sum += currentInput[j]*layers[l].Wc[j][k];
                }
                for(size_t j=0; j<hidSize; j++){
                    f_sum += layers[l].hiddenState[j]*layers[l].Uf[j][k];
                    i_sum += layers[l].hiddenState[j]*layers[l].Ui[j][k];
                    o_sum += layers[l].hiddenState[j]*layers[l].Uo[j][k];
                    c_sum += layers[l].hiddenState[j]*layers[l].Uc[j][k];
                }

                f[k] = sigmoid(f_sum);
                i[k] = sigmoid(i_sum);
                o[k] = sigmoid(o_sum);
                c_tilde[k] = tanh(c_sum);

                newCell[k] = f[k]*layers[l].cellState[k] + i[k]*c_tilde[k];
                newHidden[k] = o[k]*tanh(newCell[k]);
            }

            layers[l].hiddenState = newHidden;
            layers[l].cellState = newCell;
            currentInput = newHidden;
        }

        // выход
        for(size_t i=0; i<outputLayer.size(); i++){
            double sum=0;
            for(size_t j=0; j<layers.back().hiddenState.size(); j++){
                sum += layers.back().hiddenState[j]*outputWeights[j][i];
            }
            outputLayer[i] = sigmoid(sum);
        }
        return outputLayer;
    }

    void backpropagation(const std::vector<double>& target, const std::vector<double>& input){
        // пока реализован только выходной слой
        for(int j=0; j<outputLayer.size(); j++){
            double error = target[j] - outputLayer[j];
            double delta = error * sigmoidDerivatativ(outputLayer[j]);
            for(int i=0; i<layers.back().hiddenState.size(); i++){
                outputWeights[i][j] += learningRate * delta * layers.back().hiddenState[i];
            }
        }
        // полноценный BPTT по всем слоям можно дописать аналогично, разворачивая ошибки назад
    }

    void train(const std::vector<double>& input, const std::vector<double>& target){
        forward(input);
        backpropagation(target, input);
    }
};

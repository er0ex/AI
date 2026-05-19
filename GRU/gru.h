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

class GRU {
private:
    std::vector<int> layerSizes; // {input, hidden1, hidden2, ..., output}
    double learningRate;

    std::mt19937 generator;
    std::uniform_real_distribution<double> dist;

    // состояния скрытых слоёв
    std::vector<std::vector<double>> hiddenStates;

    // выходной слой
    std::vector<double> outputLayer;

    // веса для каждого скрытого слоя
    struct GRULayer {
        std::vector<std::vector<double>> Wz, Uz;
        std::vector<std::vector<double>> Wr, Ur;
        std::vector<std::vector<double>> Wh, Uh;
    };
    std::vector<GRULayer> gruLayers;

    // веса скрытый→выход
    std::vector<std::vector<double>> outputWeights;

    void initializeWeight(std::vector<std::vector<double>>& weight, int rows, int cols){
        weight.resize(rows, std::vector<double>(cols));
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                weight[i][j] = dist(generator);
            }
        }
    }

public:
    GRU(const std::vector<int>& sizes, double lr)
        : layerSizes(sizes), learningRate(lr),
          generator(std::random_device{}()), dist(-1.0, 1.0)
    {
        int numHiddenLayers = sizes.size() - 2; // без входа и выхода

        hiddenStates.resize(numHiddenLayers);
        gruLayers.resize(numHiddenLayers);

        // инициализация скрытых слоёв
        for(int l = 0; l < numHiddenLayers; l++){
            int inSize = (l == 0 ? sizes[0] : sizes[l]);
            int hidSize = sizes[l+1];

            hiddenStates[l].resize(hidSize, 0.0);

            initializeWeight(gruLayers[l].Wz, inSize, hidSize);
            initializeWeight(gruLayers[l].Uz, hidSize, hidSize);

            initializeWeight(gruLayers[l].Wr, inSize, hidSize);
            initializeWeight(gruLayers[l].Ur, hidSize, hidSize);

            initializeWeight(gruLayers[l].Wh, inSize, hidSize);
            initializeWeight(gruLayers[l].Uh, hidSize, hidSize);
        }

        // выходной слой
        int lastHiddenSize = sizes[sizes.size()-2];
        int outSize = sizes.back();
        outputLayer.resize(outSize, 0.0);
        initializeWeight(outputWeights, lastHiddenSize, outSize);
    }

    // прямое распространение
    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> currentInput = input;

        for(size_t l = 0; l < gruLayers.size(); l++){
            int hidSize = hiddenStates[l].size();
            std::vector<double> z(hidSize), r(hidSize), h_tilde(hidSize), newHidden(hidSize);

            for(int i = 0; i < hidSize; i++){
                double z_sum = 0.0, r_sum = 0.0, h_sum = 0.0;

                for(size_t j = 0; j < currentInput.size(); j++) z_sum += currentInput[j] * gruLayers[l].Wz[j][i];
                for(size_t j = 0; j < hidSize; j++) z_sum += hiddenStates[l][j] * gruLayers[l].Uz[j][i];
                z[i] = sigmoid(z_sum);

                for(size_t j = 0; j < currentInput.size(); j++) r_sum += currentInput[j] * gruLayers[l].Wr[j][i];
                for(size_t j = 0; j < hidSize; j++) r_sum += hiddenStates[l][j] * gruLayers[l].Ur[j][i];
                r[i] = sigmoid(r_sum);

                for(size_t j = 0; j < currentInput.size(); j++) h_sum += currentInput[j] * gruLayers[l].Wh[j][i];
                for(size_t j = 0; j < hidSize; j++) h_sum += (hiddenStates[l][j] * r[i]) * gruLayers[l].Uh[j][i];
                h_tilde[i] = tanh(h_sum);

                newHidden[i] = (1 - z[i]) * hiddenStates[l][i] + z[i] * h_tilde[i];
            }

            hiddenStates[l] = newHidden;
            currentInput = newHidden; // выход текущего слоя идёт на вход следующего
        }

        // скрытый→выход
        for(size_t i = 0; i < outputLayer.size(); i++){
            double sum = 0.0;
            for(size_t j = 0; j < hiddenStates.back().size(); j++){
                sum += hiddenStates.back()[j] * outputWeights[j][i];
            }
            outputLayer[i] = sigmoid(sum);
        }
        return outputLayer;
    }

    void backpropagation(const std::vector<double>& target){
        // минимальная версия: обновляем только выходные веса
        for(size_t j = 0; j < outputLayer.size(); j++){
            double error = target[j] - outputLayer[j];
            double delta = error * sigmoidDerivatativ(outputLayer[j]);
            for(size_t i = 0; i < hiddenStates.back().size(); i++){
                outputWeights[i][j] += learningRate * delta * hiddenStates.back()[i];
            }
        }
    }

    void train(const std::vector<double>& input, const std::vector<double>& target){
        forward(input);
        backpropagation(target);
    }
};

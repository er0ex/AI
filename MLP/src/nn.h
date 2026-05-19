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

class NN {
private:
    //std::vector<std::vector<double>> weightsInputHidden;    //Веса между входным и скрытыми слоями
    //std::vector<std::vector<double>> weihgtsOutputHidden;   //Веса между скрытыми и выходынми слоями

    std::vector<std::vector<std::vector<double>>> weights; // weights[layer][from][to]
    std::vector<std::vector<double>> layers;               // activations[layer][neuron]

    
    std::vector<double> hiddenLayer;                        // Выходывыходного слоя
    std::vector<double> outputLayer;                        // Выходы сети

    double learningRate;
    std::mt19937 generator;
    std::uniform_real_distribution<double> dist;            //Распределение для случайных чисел

    void initializeWeight(std::vector<std::vector<double>>& weight, int rows, int cols){
        weight.resize(rows, std::vector<double>(cols));
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                weight[i][j] = dist(generator); // нужно заполнить!
            }
        }
    }
    

public:
    NN(const std::vector<int>& layerSizes, double lr) : learningRate(lr), generator(std::random_device{}()), dist(-1.0, 1.0) {
        // layerSizes = {input, hidden1, hidden2, ..., output}
        layers.resize(layerSizes.size());
        for(size_t l = 0; l < layerSizes.size(); l++){
            layers[l].resize(layerSizes[l], 0.0);
        }
        weights.resize(layerSizes.size()-1);
        for(size_t l = 0; l+1 < layerSizes.size(); l++){
            initializeWeight(weights[l], layerSizes[l], layerSizes[l+1]);
        }
    }

    //Прямое распространение
    std::vector<double> forward(const std::vector<double>& input) {
        layers[0] = input;
        for(size_t l = 1; l < layers.size(); l++){
            for(size_t j = 0; j < layers[l].size(); j++){
                double sum = 0.0;
                for(size_t i = 0; i < layers[l-1].size(); i++){
                    sum += layers[l-1][i] * weights[l-1][i][j];
                }
                layers[l][j] = sigmoid(sum);
            }
        }
        return layers.back();
    }
    

    void backpropagation(const std::vector<double>& target){
        std::vector<std::vector<double>> deltas(layers.size());
    
        // ошибка выхода
        deltas.back().resize(layers.back().size());
        for(size_t j = 0; j < layers.back().size(); j++){
            double error = target[j] - layers.back()[j];
            deltas.back()[j] = error * sigmoidDerivatativ(layers.back()[j]);
        }
    
        // ошибки скрытых слоёв
        for(int l = (int)layers.size()-2; l > 0; l--){
            deltas[l].resize(layers[l].size());
            for(size_t i = 0; i < layers[l].size(); i++){
                double error = 0.0;
                for(size_t j = 0; j < layers[l+1].size(); j++){
                    error += deltas[l+1][j] * weights[l][i][j];
                }
                deltas[l][i] = error * sigmoidDerivatativ(layers[l][i]);
            }
        }
    
        // обновление весов
        for(size_t l = 0; l+1 < layers.size(); l++){
            for(size_t i = 0; i < layers[l].size(); i++){
                for(size_t j = 0; j < layers[l+1].size(); j++){
                    weights[l][i][j] += learningRate * deltas[l+1][j] * layers[l][i];
                }
            }
        }
    }
    

    void train(const std::vector<double>& input, const std::vector<double>& target){
        forward(input);              // прямой проход
        backpropagation(target);     // обратный проход (только target)
    }    
};

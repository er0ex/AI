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

class RNN {
private:
    std::vector<double> hiddenState;    

    std::vector<std::vector<std::vector<double>>> weights; // weights[layer][from][to]
    std::vector<std::vector<double>> layers;               // activations[layer][neuron]

    std::vector<std::vector<double>> weightHiddenHidden;
    
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
    RNN(const std::vector<int>& layerSizes, double lr) 
    : learningRate(lr), generator(std::random_device{}()), dist(-1.0, 1.0) {
    
        // активации для всех слоёв
        layers.resize(layerSizes.size());
        for(size_t l = 0; l < layerSizes.size(); l++){
            layers[l].resize(layerSizes[l], 0.0);
        }

        // веса между слоями
        weights.resize(layerSizes.size()-1);
        for(size_t l = 0; l+1 < layerSizes.size(); l++){
            initializeWeight(weights[l], layerSizes[l], layerSizes[l+1]);
        }

        // скрытый и выходной слой
        hiddenLayer.resize(layerSizes[1], 0.0);
        outputLayer.resize(layerSizes.back(), 0.0);

        // состояние скрытого слоя
        hiddenState.resize(layerSizes[1], 0.0);

        // веса скрытый→скрытый
        initializeWeight(weightHiddenHidden, layerSizes[1], layerSizes[1]);
    }


    //Прямое распространение
    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> newHidden(hiddenState.size());
    
        // вход → скрытый + скрытый → скрытый
        for(size_t i = 0; i < hiddenState.size(); i++){
            double sum = 0.0;
            for(size_t j = 0; j < input.size(); j++){
                sum += input[j] * weights[0][j][i];
            }
            for(size_t j = 0; j < hiddenState.size(); j++){
                sum += hiddenState[j] * weightHiddenHidden[j][i];
            }
            newHidden[i] = sigmoid(sum);
        }
        hiddenState = newHidden;
    
        // скрытый → выход
        for(size_t i = 0; i < outputLayer.size(); i++){
            double sum = 0.0;
            for(size_t j = 0; j < hiddenState.size(); j++){
                sum += hiddenState[j] * weights.back()[j][i];
            }
            outputLayer[i] = sigmoid(sum);
        }
        return outputLayer;
    }
    
    
    

    void backpropagation(const std::vector<double>& target){
        std::vector<double> outputError(outputLayer.size());
        std::vector<double> hiddenError(hiddenState.size());
    
        // ошибка выхода
        for(size_t j = 0; j < outputLayer.size(); j++){
            outputError[j] = target[j] - outputLayer[j];
            double delta = outputError[j] * sigmoidDerivatativ(outputLayer[j]);
            for(size_t i = 0; i < hiddenState.size(); i++){
                weights.back()[i][j] += learningRate * delta * hiddenState[i];
            }
        }
    
        // ошибка скрытого состояния
        for(size_t i = 0; i < hiddenState.size(); i++){
            double error = 0.0;
            for(size_t j = 0; j < outputLayer.size(); j++){
                error += outputError[j] * weights.back()[i][j];
            }
            hiddenError[i] = error * sigmoidDerivatativ(hiddenState[i]);
        }
    
        // обновление весов вход→скрытый
        for(size_t i = 0; i < layers[0].size(); i++){
            for(size_t j = 0; j < hiddenState.size(); j++){
                weights[0][i][j] += learningRate * hiddenError[j] * layers[0][i];
            }
        }
    
        // обновление весов скрытый→скрытый
        for(size_t i = 0; i < hiddenState.size(); i++){
            for(size_t j = 0; j < hiddenState.size(); j++){
                weightHiddenHidden[i][j] += learningRate * hiddenError[j] * hiddenState[i];
            }
        }
    }
    
    

    void train(const std::vector<double>& input, const std::vector<double>& target){
        layers[0] = input;          // сохраняем вход
        forward(input);             // прямой проход
        backpropagation(target);    // обратный проход
    }
     
};

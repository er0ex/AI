#include "nn.h"
#include <iostream>

int main() {
    //вход. скрытыt слоb. выход. lr 
    NN nn({2, 3, 4, 1}, 0.1); 


    std::vector<std::vector<double>> trainingInputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<std::vector<double>> trainingOutputs = {
        {0},
        {0},
        {0},
        {1}
    };

    for(int epoch = 0; epoch < 10000; epoch++){
        for(size_t i = 0; i < trainingInputs.size(); i++){
            nn.train(trainingInputs[i], trainingOutputs[i]);
        }

        if(epoch % 1000 == 0){
            std::cout << "Epoch: " << epoch;
            for(const auto& input : trainingInputs){
                auto output = nn.forward(input);
                std::cout << "Input: [" << input[0] <<", "<< input[1] <<"] ==> Output: "<< output[0]; 
            }
            std::cout << std::endl;
        }
    }

    std::cout << "Final Result" << std::endl;
    for(const auto& input : trainingInputs){
        auto output = nn.forward(input);
        std::cout << "Input: [" << input[0] <<", "<< input[1] <<"] ==> Output: "<< output[0]; 
    }

    return 0;
}
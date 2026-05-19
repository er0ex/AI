#include "rnn.h"
#include <iostream>

int main() {
    RNN rnn({2, 4, 1}, 0.01); // 2 входа, скрытый слой на 4 нейрона, 1 выход

    std::vector<std::vector<double>> sequence = {
        {0,1}, {1,0}, {1,1}, {0,0}
    };

    std::vector<std::vector<double>> targets = {
        {1}, {1}, {0}, {0} // пример: XOR или другая задача
    };

    for(int epoch = 0; epoch < 100000; epoch++){
        for(size_t i = 0; i < sequence.size(); i++){
            rnn.train(sequence[i], targets[i]);
        }
    }

    for(auto& step : sequence){
        auto out = rnn.forward(step);
        std::cout << "Input: [" << step[0] << "," << step[1] << "] Output: " << out[0] << std::endl;
    }

}
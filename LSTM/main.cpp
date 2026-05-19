#include "lstm.h"
#include <iostream>
#include <map>
#include <string>
#include <algorithm>

std::vector<double> oneHot(int index, int vocabSize){
    std::vector<double> v(vocabSize, 0.0);
    v[index] = 1.0;
    return v;
}

int main() {
    std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::map<char,int> alphabet;
    for(size_t i=0; i<chars.size(); i++) alphabet[chars[i]] = i;

    std::string text = "HELLO";

    std::vector<std::vector<double>> inputs, targets;
    for(size_t i=0; i+1<text.size(); i++){
        inputs.push_back(oneHot(alphabet[text[i]], chars.size()));
        targets.push_back(oneHot(alphabet[text[i+1]], chars.size()));
    }

    // создаём LSTM: вход=26, скрытый=32, выход=26
    LSTM lstm({26, 32, 32, 26}, 0.01);

    for(int epoch=0; epoch<10000; epoch++){
        for(size_t i=0; i<inputs.size(); i++){
            lstm.train(inputs[i], targets[i]);
        }
    }

    char s;
    std::cout << "Введите букву (A-Z): ";
    std::cin >> s;
    s = toupper(s);

    if(alphabet.find(s) == alphabet.end()){
        std::cout << "Такой буквы нет в алфавите!" << std::endl;
        return 0;
    }

    auto out = lstm.forward(oneHot(alphabet[s], chars.size()));
    int predicted = std::distance(out.begin(), std::max_element(out.begin(), out.end()));
    std::cout << "Предсказанная следующая буква после " << s << ": " << chars[predicted] << std::endl;

    return 0;
}

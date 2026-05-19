#include "gru.h"
#include <iostream>
#include <map>
#include <string>
#include <algorithm>

// функция для one-hot кодировки
std::vector<double> oneHot(int index, int vocabSize){
    std::vector<double> v(vocabSize, 0.0);
    v[index] = 1.0;
    return v;
}

int main() {
    // алфавит
    std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::map<char,int> alphabet;
    for(size_t i=0; i<chars.size(); i++) alphabet[chars[i]] = i;

    // текст для обучения
    std::string text = "HELLO";

    // формируем обучающие пары (символ -> следующий символ)
    std::vector<std::vector<double>> inputs, targets;
    for(size_t i=0; i+1<text.size(); i++){
        inputs.push_back(oneHot(alphabet[text[i]], chars.size()));
        targets.push_back(oneHot(alphabet[text[i+1]], chars.size()));
    }

    // создаём GRU: вход=26, скрытый=32, выход=26
    GRU rnn({26, 32, 26}, 0.01);

    // обучение
    for(int epoch=0; epoch<10000; epoch++){
        for(size_t i=0; i<inputs.size(); i++){
            rnn.train(inputs[i], targets[i]);
        }
    }

     // ввод буквы пользователем
     char s;
     std::cout << "Введите букву (A-Z): ";
     std::cin >> s;
     s = toupper(s); // переводим в верхний регистр
 
     if(alphabet.find(s) == alphabet.end()){
         std::cout << "Такой буквы нет в алфавите!" << std::endl;
         return 0;
     }
 
     // предсказание
     auto out = rnn.forward(oneHot(alphabet[s], chars.size()));
     int predicted = std::distance(out.begin(), std::max_element(out.begin(), out.end()));
     std::cout << "Предсказанная следующая буква после " << s << ": " << chars[predicted] << std::endl;
 
     return 0;
}

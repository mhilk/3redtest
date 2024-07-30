//
// Created by matt on 28-Jul-24.
//

#include <cassert>
#include <fstream>
#include <iostream>

#include "Book.h"
#include "OrderParser.h"

template<int levels, char prefix>
void printLevelHeader(std::ofstream &output) {
    for (int i = 0; i < levels; ++i) {
        output <<',' <<prefix<< 'p' << i << ',' <<prefix<< 'q' << i;
    }
}

int main(int argc, char *argv[]) {
    assert(argc == 3);
    std::ifstream input{argv[1]};
    std::ofstream output{argv[2], std::ios::out | std::ios::trunc};
    std::string line;
    OrderUpdate update{};
    Book book{};
    std::getline(input, line);
    std::cout << line << std::endl;
    assert(line == "timestamp,side,action,id,price,quantity");
    constexpr int printedLevels = 5;
    output << "timestamp,price,side";
    printLevelHeader<5, 'b'>(output);
    printLevelHeader<5, 'a'>(output);
    output << '\n';

    while (input.is_open()) {
        if (input.eof()) {
            break;
        }
        if (input.good()) {
            std::getline(input, line);
            if (line.empty()) {
                break;
            }
            update.read(line);
            // std::cout << update;
            book.handleUpdate(update);
            book.printCsv<5>(output, update);
            output << "\n";
        } else {
            throw std::runtime_error("fail");
        }

    }
    input.close();
    output.close();
}

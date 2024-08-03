//
// Created by matt on 28-Jul-24.
//

#ifndef ORDERPARSER_H
#define ORDERPARSER_H
#include <istream>
#include <sstream>
#include <cassert>


enum class Side {
    Buy,
    Sell
};

enum class Action {
    Add,
    Delete,
    Modify
};

inline std::istream &operator>>(std::istream &in, Side &rhs) {
    char side;
    in >> side;
    switch (side) {
        case ('b'):
            rhs = Side::Buy;
            break;
        case ('a'):
            rhs = Side::Sell;
            break;
        default:
            throw std::runtime_error(std::string(1, side));
    }
    return in;
}

inline std::istream &operator>>(std::istream &in, Action &rhs) {
    char action;
    in >> action;
    switch (action) {
        case ('a'):
            rhs = Action::Add;
            break;
        case ('d'):
            rhs = Action::Delete;
            break;
        case ('m'):
            rhs = Action::Modify;
            break;
        default:
            throw std::runtime_error(std::string(1, action));
    }
    return in;
}

struct OrderUpdate {
    long timestamp;
    Side side;
    Action action;
    long id;
    long price;
    long quantity;

    static void read_and_check_comma(std::istringstream &in) {
        char comma;
        in >> comma;
        assert(comma == ',');
    }

    void read(const std::string &line) {
        std::istringstream in(line);
        in >> this->timestamp;
        read_and_check_comma(in);
        in >> this->side;
        read_and_check_comma(in);
        in >> this->action;
        read_and_check_comma(in);
        in >> this->id;
        read_and_check_comma(in);
        in >> this->price;
        read_and_check_comma(in);
        in >> this->quantity;
    }
};

inline std::ostream &operator<<(std::ostream &lhs, const OrderUpdate &rhs) {
    lhs << rhs.timestamp << "," << rhs.id << std::endl;
    return lhs;
}

#endif //ORDERPARSER_H

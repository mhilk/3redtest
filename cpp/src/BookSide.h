//
// Created by matt on 28-Jul-24.
//

#ifndef BOOKSIDE_H
#define BOOKSIDE_H
#include <cassert>
#include <map>

#include "BookLevel.h"
#include "BookOrder.h"
#include "OrderParser.h"


template<typename comp>
class BookSide {
public:
    BookSide() = default;

    void addOrder(long price, long entry_time, long qty) {
        auto pair = levels.find(price);
        if (pair == levels.end()) {
            levels.emplace(price, BookLevel{entry_time, qty});
        } else {
            pair->second.add(entry_time, qty);
        }
    }

    void removeOrder(long price, long timestamp, long qty) {
        auto pair = levels.find(price);
        assert(pair != levels.end());
        BookLevel &book_level = pair->second;
        book_level.remove(timestamp, qty);
        if (book_level.empty()) {
            levels.erase(pair);
        }
    }

    void updateOrder(long price, long timestamp, long delta_qty) {
        auto pair = levels.find(price);
        assert(pair != levels.end());
        pair->second.update(timestamp, delta_qty);
    }

    template<int nLevels>
    void printTopN(std::ostream &os) const {
        auto begin = levels.begin();
        int i = 0;
        for (; i < nLevels && begin != levels.end(); ++i) {
            os << ',' << begin->first << ',' << begin->second.getQty();
            ++begin;
        }
        for (; i < nLevels; ++i) {
            os << ',' << 0 << ',' << 0;
        }
    }

private:
    std::map<long, BookLevel, comp> levels;
};


#endif //BOOKSIDE_H

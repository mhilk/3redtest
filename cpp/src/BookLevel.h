//
// Created by matt on 28-Jul-24.
//

#ifndef BOOKLEVEL_H
#define BOOKLEVEL_H
#include <vector>

#include "BookOrder.h"


class BookLevel {
public:
    explicit BookLevel(long creation_time, long qty)
        : qty(qty),
          order_count(1),
          last_update_time(creation_time),
          creation_time(creation_time) {
    }

    void add(long entry_time, long qty) {
        this->qty += qty;
        ++this->order_count;
        this->last_update_time = entry_time;
    }

    void remove(long timestamp, long qty) {
        --this->order_count;
        if (this->order_count) {
            this->qty -= qty;
            this->last_update_time = timestamp;
        }
    }

    void update(long timestamp, long delta_qty) {
        this->qty += delta_qty;
        this->last_update_time = timestamp;
    }

    [[nodiscard]] bool empty() const {
        return order_count == 0;
    }

    [[nodiscard]] long getQty() const {
        return qty;
    }

private:
    long qty;
    long order_count;
    long last_update_time;
    long creation_time;
};


#endif //BOOKLEVEL_H

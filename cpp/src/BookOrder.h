//
// Created by matt on 28-Jul-24.
//

#ifndef BOOKORDER_H
#define BOOKORDER_H
#include "OrderParser.h"

enum class UpdateType {
    PriceIn,
    QtyUp,
    QtyDown,
    PriceOut
};

class BookOrder {
public:
    BookOrder(long entry_time, long price, long quantity, Side side): entryTime(entry_time),
                                                                      price(price),
                                                                      nUpdates(0),
                                                                      qty(quantity),
                                                                      lastUpdate(UpdateType::PriceIn),
                                                                      side(side) {
    }

    [[nodiscard]] long getQty() const {
        return qty;
    }

    [[nodiscard]] long getPrice() const {
        return price;
    }

    [[nodiscard]] Side getSide() const {
        return side;
    }

    void update(long price, long quantity) {
        long priceChange = price - this->price;
        if (side == Side::Sell) {
            priceChange = -priceChange;
        }
        if (priceChange == 0) {
            if (quantity < qty) {
                lastUpdate = UpdateType::QtyDown;
            } else {
                lastUpdate = UpdateType::QtyUp;
            }
        } else {
            if (priceChange < 0) {
                lastUpdate = UpdateType::PriceOut;
            } else {
                lastUpdate = UpdateType::PriceIn;
            }
        }
        this->price = price;
        this->qty = quantity;
    }

private:
    const long entryTime;
    long price;
    long nUpdates;
    long qty;
    UpdateType lastUpdate;
    const Side side;
};


#endif //BOOKORDER_H

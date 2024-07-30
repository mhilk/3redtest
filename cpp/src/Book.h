//
// Created by matt on 28-Jul-24.
//

#ifndef BOOK_H
#define BOOK_H
#include <cassert>
#include <unordered_map>

#include "BookOrder.h"
#include "BookSide.h"
#include "OrderParser.h"


inline std::ostream &operator<<(std::ostream &lhs, Side rhs) {
    switch (rhs) {
        case Side::Buy:
            lhs << 'b';
            break;
        case Side::Sell:
            lhs << 'a';
            break;
    }
    return lhs;
}

class Book {
public:
    Book() = default;

    template<Side side, typename comp>
    void internal_handle_update(const OrderUpdate &order_update) {
        switch (order_update.action) {
            case Action::Add:
                add_order<side, comp>(order_update);
                break;
            case Action::Delete:
                delete_order<side, comp>(order_update);
                break;
            case Action::Modify:
                modify_order<side, comp>(order_update);
                break;
            default:
                throw std::runtime_error("unknown action");
        }
    }

    void handleUpdate(const OrderUpdate &order_update) {
        switch (order_update.side) {
            case Side::Buy:
                internal_handle_update<Side::Buy, std::greater<> >(order_update);
                break;
            case Side::Sell:
                internal_handle_update<Side::Sell, std::less<> >(order_update);
                break;
        }
    }

    template<int nLevels>
    void printCsv(std::ostream &output, const OrderUpdate &order_update) const {
        output << order_update.timestamp << ',' << order_update.price << ',' << order_update.side;
        bids.printTopN<nLevels>(output);
        offs.printTopN<nLevels>(output);
    }

private:
    BookSide<std::greater<> > bids;
    BookSide<std::less<> > offs;

    std::unordered_map<long, BookOrder *> orders;

    template<Side side, typename comp>
    BookSide<comp> &get_side() {
        if constexpr (side == Side::Buy) {
            return bids;
        }
        if constexpr (side == Side::Sell) {
            return offs;
        }
        static_assert("unknown side");
    }

    template<Side side, typename comp>
    void add_order(const OrderUpdate &order_update) {
        BookSide<comp> &book_side = get_side<side, comp>();
        orders[order_update.id] = new BookOrder{
            order_update.timestamp, order_update.price, order_update.quantity, order_update.side
        };
        book_side.addOrder(order_update.price, order_update.timestamp, order_update.quantity);
    }

    template<Side side, typename comp>
    void delete_order(const OrderUpdate &order_update) {
        BookSide<comp> &book_side = get_side<side, comp>();
        auto book_order = orders.find(order_update.id);
        assert(book_order != orders.end());
        orders.erase(book_order);
        BookOrder *existing_order = book_order->second;
        book_side.removeOrder(existing_order->getPrice(), order_update.timestamp, existing_order->getQty());
    }

    template<Side side, typename comp>
    void modify_order(const OrderUpdate &order_update) {
        BookSide<comp> &book_side = get_side<side, comp>();
        auto book_order = orders.find(order_update.id);
        assert(book_order != orders.end());
        BookOrder *existing_order = book_order->second;
        assert(existing_order->getSide() == order_update.side);
        if (order_update.price == existing_order->getPrice()) {
            book_side.updateOrder(order_update.price, order_update.timestamp,
                                  order_update.quantity - existing_order->getQty());
        } else {
            book_side.removeOrder(existing_order->getPrice(), order_update.timestamp, existing_order->getQty());
            book_side.addOrder(order_update.price, order_update.timestamp, order_update.quantity);
        }
    }
};


#endif //BOOK_H

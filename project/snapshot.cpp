#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>

using namespace std;

// Define constants
const char ADD = 'a';
const char DELETE = 'd';
const char MODIFY = 'm';
const char BUY = 'b';
const char SELL = 'a';


typedef struct MessageType {
    long timestamp;
    int side;
    char action;
    long id;
    long price;
    long quantity;
} MessageType;


class DepthSnapshot
{
private:
    map<int, MessageType> messages_;
    map<int, int> bids_;
    map<int, int> asks_;
public:
    // Update Methods
    void update(const MessageType& message)
    {
        map<int, int>* snapshot;
        if (message.side == BUY)
        {
            snapshot = &bids_;
        }
        else
        {
            snapshot = &asks_;        
        }
        if(message.action == ADD)
        {
            messages_[message.id] = message;
            if(snapshot->find(message.price) != snapshot->end())
            {
                (*snapshot)[message.price] += message.quantity;
            }
            else
            {
                (*snapshot)[message.price] = message.quantity;
            }
        }
        else if(message.action == DELETE)
        {
            MessageType delete_msg = messages_[message.id];
            messages_.erase(message.id);
            (*snapshot)[delete_msg.price] -= delete_msg.quantity;
            if((*snapshot)[delete_msg.price] <= 0) snapshot->erase(delete_msg.price);
        }
        else if(message.action == MODIFY)
        {
            MessageType prev_msg = messages_[message.id];
            messages_[message.id] = message;
            if(message.price == prev_msg.price)
            {
                int diff = message.quantity - prev_msg.quantity;
                (*snapshot)[prev_msg.price] += diff;
            }
            else
            {
                (*snapshot)[prev_msg.price] -= prev_msg.quantity;
                if(snapshot->find(message.price) != snapshot->end())
                {
                    (*snapshot)[message.price] += message.quantity;
                }
                else
                {
                    (*snapshot)[message.price] = message.quantity;
                }
            }
            if((*snapshot)[prev_msg.price] <= 0) snapshot->erase(prev_msg.price);
        }
        else
        {
            // Do nothing for unkonwn action
        }
    }
    // Get Current Snapshot
    map<int, int> getSnapshot(char side)
    {
        if(side == BUY)
        {
            return bids_;
        }
        else if(side == SELL)
        {
            return asks_;
        }
        else
        {
            map<int, int> garbage;
            return garbage;
        }
    }
    // Clera status
    void clear()
    {
        bids_.clear();
        asks_.clear();
        messages_.clear();
    }
    // Constructor
    DepthSnapshot() {};
};


int main(int argc, char* argv[])
{
    DepthSnapshot depth = DepthSnapshot();
    string output_dir = "outputs/";
    for(int i = 1; i < argc; ++i)
    {
        string line, word;
        fstream fout;
        fstream fin;
        string path = argv[i];
        fin.open(path, ios::in);
        string filename;
        stringstream path_ss(path);
        // Last word after splitting path is the input file name
        while(getline(path_ss, filename, '/')) {};
        string output_filename = output_dir + "output_" + filename;
        fout.open(output_filename, ios::out);
        // Header, add quantity and action for debugging purpose
        fout << "timestamp" << "," << "price" << "," << "side" << "," << "quantity" << "," << "action";
        for(int i = 0; i < 5; ++i)
        {
            string str_i = to_string(i);
            fout << "," << "bp" + str_i << "," << "bq" + str_i;
        }
        for(int i = 0; i < 5; ++i)
        {
            string str_i = to_string(i);
            fout << "," << "ap" + str_i << "," << "aq" + str_i;
        }
        // Addtional Feature, Order Flow Imbalance
        fout << "," << "ofi";
        fout << "\n";
        depth.clear();
        int count = 0;
        // parameter for Order Flow Imbalance
        bool bid_ready = false;
        bool ask_ready = false;
        int best_bid_price, best_bid_volume, best_ask_price, best_ask_volume;
        // Get rid of header from input
        getline(fin, line);
        while(getline(fin, line))
        {
            stringstream ss(line);
            vector<string> words;
            count++;
            while(getline(ss, word, ','))
            {
                words.push_back(word);
            }
            // Update Snapshot
            MessageType message;
            long timestamp;
            istringstream(words[0]) >> timestamp;
            message.timestamp = timestamp;
            char side = words[1][0];
            message.side = side;
            char action = words[2][0];
            message.action = action;
            long id;
            istringstream(words[3]) >> id;
            message.id = id;
            long price;
            istringstream(words[4]) >> price;
            message.price = price;
            long quantity;
            istringstream(words[5]) >>  quantity;
            message.quantity = quantity;
            depth.update(message);
            // Write new snapshot
            fout << timestamp << "," << price << "," << side << "," << quantity << "," << action;
            int count;
            int bid_flow = 0;
            int ask_flow = 0;
            // Bid
            count = 0;
            // Ascending Order
            map<int, int> bids = depth.getSnapshot(BUY);
            for(auto iter = bids.rbegin(); iter != bids.rend(); ++iter)
            {
                fout << "," << iter->first << "," << iter->second;
                count++;
                if(count >= 5) break;
            }
            if(count > 0)
            {
                int new_bid_price = bids.rbegin()->first;
                int new_bid_volume = bids.rbegin()->second;
                if(bid_ready)
                {
                    if(new_bid_price > best_bid_price)
                    {
                        bid_flow = new_bid_volume;
                    }
                    else if(new_bid_price == best_bid_price)
                    {
                        bid_flow = new_bid_volume - best_bid_volume;
                    }
                    else
                    {
                        bid_flow = -best_bid_volume;
                    }
                }
                else
                {
                    bid_ready = true;
                    bid_flow = new_bid_volume;
                }
                best_bid_price = new_bid_price;
                best_bid_volume = new_bid_volume;   
            }
            for(int i = count; i < 5; ++i)
            {
                fout << "," << ",";
            }
            // Ask
            count = 0;
            // Ascending Order
            map<int, int> asks = depth.getSnapshot(SELL);
            for(auto iter = asks.begin(); iter != asks.end(); ++iter)
            {
                fout << "," << iter->first << "," << iter->second;
                count++;
                if(count >= 5) break;
            }
            if(count > 0)
            {
                int new_ask_price = asks.begin()->first;
                int new_ask_volume = asks.rbegin()->second;
                if(ask_ready)
                {
                    if(new_ask_price < best_ask_price)
                    {
                        ask_flow = new_ask_volume;
                    }
                    else if(new_ask_price == best_ask_price)
                    {
                        ask_flow = new_ask_volume - best_ask_volume;
                    }
                    else
                    {
                        ask_flow = -best_ask_volume;
                    }
                }
                else
                {
                    ask_ready = true;
                    ask_flow = new_ask_volume;
                }
                best_ask_price = new_ask_price;
                best_ask_volume = new_ask_volume;   
            }
            for(int i = count; i < 5; ++i)
            {
                fout << "," << ",";
            }
            fout << "," << bid_flow - ask_flow;
            fout << "\n";

        }
        fin.close();
        fout.close();
    }
    return 0;
}

import hashlib
from datetime import datetime
from flask import Flask
from flask import request
from copy import deepcopy
import json
import requests
node = Flask(__name__)


class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.hash_block()

    def hash_block(self):
        sha = hashlib.sha256()
        input_str = str(self.index) +\
                    str(self.timestamp) +\
                    str(self.data) +\
                    str(self.previous_hash)
        input_bytes = bytes(input_str, 'utf-8')
        sha.update(input_bytes)
        return sha.hexdigest()


def create_genesis_block():
    # Manually construct a block with
    # index zero and arbitrary previous hash
    return Block(0, datetime.now(),
                 {"proof-of-work": 9,
                  "transactions": None},
                 "0")


# A completely random address of the owner of this node
miner_address = "q3nf394hjg-random-miner-address-34nf3i4nflkn3oi"

blockchain = []
blockchain.append(create_genesis_block())
this_nodes_transactions = []
peer_nodes = []

mining = True


@node.route("/txion", methods=["POST"])
def transactions():
    new_txion = request.get_json()
    this_nodes_transactions.append(new_txion)
    print("New transaction")
    print("FROM: {}".format(new_txion["from"]))
    print("TO: {}".format(new_txion["to"]))
    print("AMOUNT: {}\n".format(new_txion["amount"]))
    return "Transaction submission successful\n"


@node.route("/blocks", methods=["GET"])
def get_blocks():
    chain_to_send = deepcopy(blockchain)
    for i in range(len(chain_to_send)):
        block = chain_to_send[i]
        block_index = str(block.index)
        block_timestamp = str(block.timestamp)
        block_data = str(block.data)
        block_hash = block.hash
        chain_to_send = {
            "index": block_index,
            "timestamp": block_timestamp,
            "data": block_data,
            "hash": block_hash
        }
    chain_to_send = json.dumps(chain_to_send)
    return chain_to_send


def find_new_chains():
    other_chains = []
    for node_url in peer_nodes:
        block = requests.get(node_url + "/blocks").content
        block = json.loads(block)
        other_chains.append(block)
    return other_chains


def consensus():
    global blockchain
    other_chains = find_new_chains()
    longest_chain = blockchain
    for chain in other_chains:
        if len(longest_chain) < len(chain):
            longest_chain = chain
    blockchain = longest_chain


def proof_of_word(last_proof):
    incrementor = last_proof + 1
    while not (incrementor % 9 == 0 and incrementor % last_proof == 0):
        incrementor += 1
    return incrementor


@node.route('/mine', methods=["Get"])
def mine():
    last_block = blockchain[-1]
    last_proof = last_block.data["proof-of-work"]
    proof = proof_of_word(last_proof)
    this_nodes_transactions.append(
        {"from": "network", "to": miner_address, "amount": 1}
    )

    new_block_data = {
        "proof-of-work": proof,
        "transactions": list(this_nodes_transactions)
    }
    new_block_index = last_block.index + 1
    new_block_timestamp = datetime.now()
    last_block_hash = last_block.hash

    this_nodes_transactions[:] = []

    mined_block = Block(
        new_block_index,
        new_block_timestamp,
        new_block_data,
        last_block_hash
    )

    blockchain.append(mined_block)

    return json.dumps({
        "index": new_block_index,
        "timestamp": str(new_block_timestamp),
        "data": new_block_data,
        "hash": last_block_hash,
    }) + "\n"


if __name__ == "__main__":
    node.run()

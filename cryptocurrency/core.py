class BaseBlockchain(object):
    def __init__(self):
        self.chain = []
        self.current_transactions = []

    def new_blcok(self):
        pass

    def new_transaction(self):
        pass

    @staticmethod
    def hash(block):
        pass

    @property
    def last_block(self):
        pass

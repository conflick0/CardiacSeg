from abc import ABC, abstractmethod


class Transform(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def get_train_transform(self):
        pass

    @abstractmethod
    def get_val_transform(self):
        pass

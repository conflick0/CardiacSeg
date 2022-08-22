from abc import ABC, abstractmethod

class Dataset(ABC):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_data_dicts(self):
        img_pths, lbl_pths = self.get_data_paths()
        data_dicts = [
            {"image": img_pth, "label": lbl_pth}
            for img_pth, lbl_pth in zip(img_pths, lbl_pths)
        ]
        return data_dicts

    @abstractmethod
    def get_data_paths(self):
        pass
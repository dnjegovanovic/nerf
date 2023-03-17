from torch.utils.data import Dataset
from pathlib import Path

from torch.utils.data.dataset import Dataset

class BaseDataset(Dataset):
    def __init__(self, root_dir, base_name, extension):
        self.dataset_id = hash(str(root_dir) + "_" + extension)
        self.root_dir = Path(root_dir)
        self.base_name = base_name
        self.extension = extension

        self.data_path = root_dir + base_name + extension

    def __getitem__(self, index):
        pass

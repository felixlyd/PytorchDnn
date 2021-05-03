import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from common import TRAIN, VALID, TEST


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.classes = []
        self.inputs = []
        self.labels = []

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]

    def __len__(self):
        return len(self.inputs)


class DataIter:
    def __init__(self, dataloader, device):
        self.dataloader = [(x, y) for x, y in tqdm(dataloader)]
        self.index = 0
        self.len = len(self.dataloader)
        self.device = device

    def _to_tensor(self, x, y):
        return x.to(self.device), y.to(self.device)

    def __next__(self):
        if self.index >= self.len:
            self.index = 0
            raise StopIteration
        else:
            x, y = self.dataloader[self.index]
            self.index = self.index + 1
            x, y = self._to_tensor(x, y)
            return x, y

    def __iter__(self):
        return self

    def __len__(self):
        return self.len


class BaseLoader:
    def __init__(self, opt):
        self.data_dir = opt.data
        self.batch_size = opt.batch_size
        self.train_loaded = None
        self.class_nums = 0
        self.valid_loaded = None
        self.test_loaded = None
        self.thread = opt.thread_num
        self.gpu_ids = opt.gpu_ids
        self.device = self._set_device()
        self.inputs = {}
        self.is_train = (opt.test is False)

    def _load_sets(self, tag=TRAIN, **kwargs):
        print("Loading datasets...")
        data_sets = BaseDataset()
        return data_sets

    def _load_train_valid(self, **kwargs):
        print("Loading datasets...")
        train_sets = self._load_sets(tag=TRAIN, **kwargs)
        self.train_loaded = DataLoader(train_sets, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.thread)
        valid_sets = self._load_sets(tag=VALID, **kwargs)
        self.valid_loaded = DataLoader(valid_sets, batch_size=self.batch_size, shuffle=True)
        self.class_nums = len(train_sets.classes)
        print("Train sets:", len(train_sets))
        print("Valid sets:", len(valid_sets))
        print("Batch size:", self.batch_size)
        print("Class nums:", self.class_nums)

    def _load_test(self, **kwargs):
        test_sets = self._load_sets(tag=TEST, **kwargs)
        self.test_loaded = DataLoader(test_sets, batch_size=self.batch_size, num_workers=self.thread)

    def load(self, **kwargs):
        if self.is_train:
            self._load_train_valid(**kwargs)
            print("Handling datasets...")
            self.inputs[TRAIN] = DataIter(self.train_loaded, self.device)
            self.inputs[VALID] = DataIter(self.valid_loaded, self.device)
        else:
            self._load_test(**kwargs)
            print("Handling datasets...")
            self.inputs[TEST] = DataIter(self.test_loaded, self.device)

    def _set_device(self):
        device = "cpu"
        cuda = torch.cuda.is_available()
        if self.gpu_ids[0] != -1:
            if cuda:
                device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
            else:
                device = torch.device('cpu')
        return device

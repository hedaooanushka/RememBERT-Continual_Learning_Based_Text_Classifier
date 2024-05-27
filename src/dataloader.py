#Consists of dataloader for loading the apporpriate dataset 
from torch.utils.data import Dataset
import torch
import re
from torch.utils.data import DataLoader, random_split
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.generators import nc_benchmark
from get_dataset import *



class ASC:

    def __init__(self, seq_len, tokenizer, training_mode):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.training_mode = training_mode

    def dataset(self):
        return 0
    def load_dataset():
        return 0
    
class DSC:

    def __init__(self, seq_len, tokenizer, training_mode, n_exp):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.training_mode = training_mode
        self.n_exp = n_exp

    def get_dataset(self, mode):
        dsc_dataset = DSC_Loader()

        if mode == "TRAD":
            return dsc_dataset.get_traditional()
        elif mode == "CONT":
            train_text, train_labels, test_text, test_labels = dsc_dataset.get_continual()
            return train_text, train_labels, test_text, test_labels
        else:
            raise FileNotFoundError

    def load_dataset(self):

        if self.training_mode == "TRAD":
            dataset = DSC.get_dataset(self.training_mode)
            class MyDataset(Dataset):
                def __init__(self, dataset, tokenizer, seq_len):
                    self.dataset = dataset
                    self.tokenizer = tokenizer
                    self.seq_len = seq_len

                def __len__(self):
                    return len(self.dataset)

                @staticmethod
                def preprocess_text(text):
                    # Lowercase the text
                    text = text.lower()
                    # Remove URLs
                    text = re.sub(r'http\S+|www.\S+', ' ', text)
                    # Remove emails
                    text = re.sub(r'\S*@\S*\s?', ' ', text)
                    # Remove special characters (keeping letters, numbers, and basic punctuation)
                    text = re.sub(r'[^a-z0-9,.!? ]', ' ', text)
                    return text

                def __getitem__(self, idx):
                    item = self.dataset[idx]
                    text = MyDataset.preprocess_text(item['text'])
                    encoding = self.tokenizer(text, truncation=True, padding='max_length', return_tensors='pt', max_length=self.seq_len)
                    encoding['label'] = torch.tensor(item['label'])
                    return encoding
        
            traindata = MyDataset(dataset=dataset['train'], tokenizer=self.tokenizer)
            val_size = int(len(traindata) * 0.2)  
            train_size = len(traindata) - val_size  
            traindata, valdata = random_split(traindata, [train_size, val_size])
            testdata = MyDataset(dataset=dataset['test'], tokenizer=self.tokenizer)
            train_dataloader = DataLoader(traindata, batch_size=32, shuffle=True)
            val_dataloader = DataLoader(valdata, batch_size=32)
            test_dataloader = DataLoader(testdata, batch_size=32)
            return train_dataloader, val_dataloader, test_dataloader
        
        elif self.training_mode == "CONT":
            train_text, train_labels, test_text, test_labels = DSC.get_dataset(self.training_mode)
            class TextDataset(Dataset):
                def __init__(self, texts, labels, tokenizer, seq_len):
                    """
                    Args:
                        texts (list of str): List of text samples.
                        labels (list of int): List of labels corresponding to the text samples.
                    """
                    self.texts = texts
                    self.labels = labels
                    self.tokenizer = tokenizer
                    self.seq_len = seq_len

                def __len__(self):
                    return len(self.texts)

                @staticmethod
                def preprocess_text(text):
                    # Lowercase the text
                    text = text.lower()
                    # Remove URLs
                    text = re.sub(r'http\S+|www.\S+', ' ', text)
                    # Remove emails
                    text = re.sub(r'\S*@\S*\s?', ' ', text)
                    # Remove special characters (keeping letters, numbers, and basic punctuation)
                    text = re.sub(r'[^a-z0-9,.!? ]', ' ', text)
                    return text

                def __getitem__(self, idx):
                    text = self.texts[idx]
                    label = self.labels[idx]

                    # Tokenize the text (you could add truncation and padding as needed)
                    text = TextDataset.preprocess_text(text)
                    encoded_text = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.seq_len)
                    encoded_text = encoded_text["input_ids"]

                    return encoded_text, label
            train_data = TextDataset(train_text, train_labels)
            test_data = TextDataset(test_text, test_labels)

            avl_train_data = AvalancheDataset(train_data)
            avl_test_data = AvalancheDataset(test_data)


            avl_train_data.targets = train_labels
            avl_test_data.targets = test_labels

            benchmark = nc_benchmark(
                test_dataset=avl_test_data,  
                train_dataset=avl_train_data,
                n_experiences=self.n_exp,  
                task_labels=False  
            )

            train_stream = benchmark.train_stream
            test_stream = benchmark.test_stream
            experience = train_stream[0]

            t_label = experience.task_label
            dataset = experience.dataset

            return train_stream, test_stream


class twentyNG:
    def dataset():
        return 0
    def load_dataset():
        return 0    




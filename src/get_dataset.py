#Methods of reading and loading the datasets
from datasets import load_dataset 
from datasets_path import *
import os
import json


class TwentyNG_Loader:
    def __init__(self):
        self.dataset = load_dataset("setfit/20_newsgroups")
        return self.dataset
    
class DSC_Loader:
    def __init__(self):
        self.path = DSC_setpath.get_path()
        self.dataset = {'train':[],'test':[],'val':[]}

    def get_traditional(self):    
        idx=0
        for subdir in os.listdir(self.path):
            if subdir!='XuSemEval':
                subdir_path = os.path.join(self.path, subdir+'/asc')
                for subsubdir in os.listdir(subdir_path):
                    subsubdir_path=os.path.join(subdir_path, subsubdir)
                    train_path = os.path.join(subsubdir_path, 'train.json')
                    test_path = os.path.join(subsubdir_path, 'test.json')
                    val_path = os.path.join(subsubdir_path, 'dev.json')
                    paths=[train_path,test_path,val_path]
                    for i in range(len(paths)):
                        with open(paths[i], 'r') as f:
                            l=self.dataset[list(self.dataset.keys())[i]]
                            data = json.load(f)
                            for entry in data.values():
                                if "sentence" in entry:
                                    l.append({'text':entry["sentence"],'label':idx})
                    idx+=1
            else:
                subdir_path = os.path.join(self.path, subdir+'/asc')
                flag=False
                for subsubdir in os.listdir(subdir_path):
                    subsubdir_path=os.path.join(subdir_path, subsubdir)
                    if subsubdir=='14':
                        flag=True
                    for subsubsubdir in os.listdir(subsubdir_path):
                        subsubsubdir_path=os.path.join(subsubdir_path, subsubsubdir)
                        if subsubsubdir=='rest':
                            continue
                        train_path = os.path.join(subsubsubdir_path, 'train.json')
                        test_path = os.path.join(subsubsubdir_path, 'test.json')
                        val_path = os.path.join(subsubsubdir_path, 'dev.json')
                        paths=[train_path,test_path,val_path]
                        for i in range(len(paths)):
                            with open(paths[i], 'r') as f:
                                l=self.dataset[list(self.dataset.keys())[i]]
                                data = json.load(f)
                                for entry in data.values():
                                    if flag:
                                      if "sentence" in entry:
                                                l.append({'text':entry["sentence"],'label':idx}) 
                                    else:
                                        if entry is not None:
                                            for subentry in entry.values():
                                                if "sentence" in subentry:
                                                    l.append({'text':subentry["sentence"],'label':idx})                
                        idx+=1
                    flag=False

        return self.dataset
    

    def get_continual(self):
        categories = ['Kindle_Store', 'Movies_and_TV', 'Musical_Instruments', 'Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies', 'Sports_and_Outdoors', 'Tools_and_Home_Improvement', 'Toys_and_Games', 'Video_Games']

        dsc_train_text = []
        dsc_train_labels = []
        for i in range(0, len(categories)):
            file_path = os.path.join(self.path, categories[i])
            file_path = os.path.join(file_path, 'train.json')
            with open(file_path, 'r') as file:
                data = json.load(file)
                sentence_list = [value["sentence"] for key, value in data.items()]
                labels = [i] * len(sentence_list)
                dsc_train_text += sentence_list
                dsc_train_labels += labels


        dsc_test_text = []
        dsc_test_labels = []
        for i in range(0, len(categories)):
            file_path = os.path.join(self.path, categories[i])
            file_path = os.path.join(file_path, 'test.json')
            with open(file_path, 'r') as file:
                data = json.load(file)
                sentence_list = [value["sentence"] for key, value in data.items()]
                labels = [i] * len(sentence_list)
                dsc_test_text += sentence_list
                dsc_test_labels += labels

        train_text = dsc_train_text
        train_labels = dsc_train_labels
        test_text = dsc_test_text
        test_labels = dsc_test_labels

        return train_text, train_labels, test_text, test_labels
    
class ASC_Loader:
    def __init__(self):
        self.path = ASC_setpath.get_path()
        self.dataset = {'train':[],'test':[],'val':[]}
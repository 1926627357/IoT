import numpy as np
import os
import re
from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset, DataLoader
from config import config_info
# handle os files
class FileManager(object):
    def __init__(self,RootDir):
        self.RootDir=RootDir
        for _,_,files in os.walk(self.RootDir):
            print("[Info] Correctly get files")
            self.files=files
            break
        
        self.len=len(self.files)
        for i in range(self.len):
            self.files[i]=os.path.join(self.RootDir,self.files[i])
        random.shuffle(self.files)
        self.report()
        # print(self.files)
    def report(self):
        print("[Info] In",os.path.normpath(self.RootDir).split('/')[-1],", sum of files:",len(self))
    def getItem(self,index):
        return self.files[index if index<len(self) else len(self)]
    def __len__(self):
        return self.len
    def __iter__(self):
        self.count=0
        return self
    def __next__(self):
        if self.count<len(self):
            tempFile=self.files[self.count]
            self.count+=1
            return tempFile
        else:
            raise StopIteration()
class data_loader(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)
def normalize(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm
class LoadData(object):
    def __init__(self,FileManager):
        self.FileManager=FileManager
        
        if os.path.splitext(self.FileManager.getItem(0))[1]=='.txt':
            self.loadNpData()
            self.loadNpLabel()
        else:
            self.loadFromDumpFile()
        
    def loadNpData(self):
        matrix=np.zeros((1,3))
        tqdmbar=tqdm(total=len(self.FileManager))
        tqdmbar.set_description("Loading Data ...")
        for filename in self.FileManager:
            matrix=np.vstack((matrix,np.loadtxt(filename,dtype=np.float)))
            tqdmbar.update(1)
        matrix=matrix[1:]
        matrix=matrix.reshape(-1,125,3)
        self.new_matrix=None
        for i in range(len(matrix)):
            row = np.asarray(matrix[i, :]).T
            if self.new_matrix is None:
                self.new_matrix = np.zeros((len(matrix), 3, 125))
            self.new_matrix[i]=row
        self.new_matrix=self.new_matrix.reshape(-1,3,1,125)
        tqdmbar.close()
        print("[Info] LoadDataset Shape: ",self.new_matrix.shape)

    def parsePath(self,Path):
        PathEnd=os.path.normpath(Path).split('/')[-1]
        pattern=r'[0-9]+'
        AllMatch=re.findall(pattern,PathEnd)
        return int(AllMatch[0])

    def loadNpLabel(self):
        self.label=[]
        self.LabelMap={'1':0,'2':0,'3':0,'4':0}
        tqdmbar=tqdm(total=len(self.FileManager))
        tqdmbar.set_description("Loading Label ...")
        for filename in self.FileManager:
            TempType=self.parsePath(filename)
            self.LabelMap[str(TempType)]+=1
            self.label.append(TempType-1)
            tqdmbar.update(1)
        tqdmbar.close()

        self.label=np.array(self.label)
        # print(self.label)
        # self.label=np.eye(4)[self.label]
        # print(self.label)
        print("[Info] Label shape: ",self.label.shape)
        print(f"[Info] Label NumMap 1:{self.LabelMap['1']}  2:{self.LabelMap['2']}  3:{self.LabelMap['3']}  4:{self.LabelMap['4']}")

    def dumpTorchDataset(self,batch_size,shuffle,drop_last):
        print(f"[Info] Dump to pytorch dataset format, settings: batch size[{batch_size}], shuffle[{shuffle}], drop last[{drop_last}]")
        return DataLoader(
                            data_loader(self.new_matrix, self.label, normalize), 
                            batch_size=batch_size, 
                            shuffle=True, 
                            drop_last=True
                            )

    def dumpToFile(self,mode):
        np.save(os.path.join('/home/haiqwa/Document/hw1/dataset/dumpDataset/',mode,'label'),self.label)
        np.save(os.path.join('/home/haiqwa/Document/hw1/dataset/dumpDataset/',mode,'data'),self.new_matrix)

    def loadFromDumpFile(self):
        for filename in self.FileManager:
            if 'data' in os.path.normpath(filename).split('/')[-1]:
                self.new_matrix=np.load(filename)
            else:
                self.label=np.load(filename)

def load(batch_size=64):
    return LoadData(FileManager(config_info['linke_train'])).dumpTorchDataset(batch_size=batch_size,shuffle=True,drop_last=False),\
            LoadData(FileManager(config_info['linke_test'])).dumpTorchDataset(batch_size=batch_size,shuffle=True,drop_last=False)




if __name__ == "__main__":
    train_files=FileManager('/home/haiqwa/Document/hw1/dataset/dumpDataset/train/')

    dataset=LoadData(train_files)
    # dataset.dumpToFile('train')
    # dataset.loadNpData()
    dataset.dumpTorchDataset(batch_size=32,shuffle=True,drop_last=False)
    # print([dataset.parsePath('/home/haiqwa/Document/hw1/dataset/dataset_0512_5_linke/train_margin/acce_action_{100}_{1}_1.txt')])

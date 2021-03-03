from typing import List
import math

class MaxHeap:
    data = []
    def __init__(self, arr: List[int]) -> None:
        self.data = arr.copy()
        for i in range(math.floor(len(arr)/2)-1, -1, -1):
            self.__down(i)
    
    def insert(self, num: int):
        position = len(self.data)
        self.data.append(num)
        while position != 0 and self.data[position] > self.data[math.floor((position-1)/2)]:
            self.data[position], self.data[math.floor((position-1)/2)] = self.data[math.floor((position-1)/2)], self.data[position]
            position = math.floor((position-1)/2)
    
    def delete(self):
        self.data[0] = self.data[-1]
        self.data.pop()
        self.__down(0)
    
    def __down(self, position: int):
        if position > math.floor(len(self.data)/2)-1:
            return 
        if (position+1)*2 == len(self.data):
            if self.data[(position+1)*2-1] > self.data[position]:
                self.data[position], self.data[position*2+1] = self.data[position*2+1], self.data[position]
            return
        if self.data[(position+1)*2] >= self.data[position*2+1] and self.data[(position+1)*2] >= self.data[position]:
            self.data[position], self.data[(position+1)*2] = self.data[(position+1)*2], self.data[position]
            return self.__down((position+1)*2)
        elif self.data[position*2+1] >= self.data[(position+1)*2] and self.data[position*2+1] >= self.data[position]:
            self.data[position], self.data[position*2+1] = self.data[position*2+1], self.data[position]
            return self.__down(position*2+1)

class MinHeap:
    data = []
    def __init__(self, arr: List[int]) -> None:
        self.data = arr.copy()
        for i in range(math.floor(len(arr)/2)-1, -1, -1):
            self.__down(i)
    
    def insert(self, num: int):
        position = len(self.data)
        self.data.append(num)
        while position != 0 and self.data[position] < self.data[math.floor((position-1)/2)]:
            self.data[position], self.data[math.floor((position-1)/2)] = self.data[math.floor((position-1)/2)], self.data[position]
            position = math.floor((position-1)/2)
    
    def delete(self):
        self.data[0] = self.data[-1]
        self.data.pop()
        self.__down(0)

    def __down(self, position: int):
        if position > math.floor(len(self.data)/2)-1:
            return
        if (position+1)*2 == len(self.data):
            if self.data[(position+1)*2-1] < self.data[position]:
                self.data[position], self.data[position*2+1] = self.data[position*2+1], self.data[position]
            return
        if self.data[(position+1)*2] <= self.data[position*2+1] and self.data[(position+1)*2] <= self.data[position]:
            self.data[position], self.data[(position+1)*2] = self.data[(position+1)*2], self.data[position]
            return self.__down((position+1)*2)
        elif self.data[position*2+1] <= self.data[(position+1)*2] and self.data[position*2+1] <= self.data[position]:
            self.data[position], self.data[position*2+1] = self.data[position*2+1], self.data[position]
            return self.__down(position*2+1)

            
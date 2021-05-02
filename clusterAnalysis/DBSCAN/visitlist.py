# -*- coding: UTF-8 -*-

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/3/16 11:55
  Software: PyCharm
  Profile: # visitlist类用于记录访问列表 # unvisitedlist记录未访问过的点 # visitedlist记录已访问过的点
  # unvisitednum记录访问过的点数量
"""


class visitlist:
    def __init__(self, count=0):
        self.unvisitedlist = [i for i in range(count)]
        self.visitedlist = list()
        self.unvisitednum = count

    def visit(self, pointId):
        self.visitedlist.append(pointId)
        self.unvisitedlist.remove(pointId)
        self.unvisitednum -= 1

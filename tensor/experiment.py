import numpy as np
import importlib  

from .evaluation import *

class Experiment(object):

  def __init__(self, train, test, entities, relations, param):

    self.train = train
    self.test = test

    self.entities = entities
    self.relations = relations

    self.model = getattr(importlib.import_module(".models", package="tensor"), param.model)()
    self.scorer = Scorer(train, test)
    self.param = param

  def evaluation(self):
    self.model.fit(self.train, len(self.entities), len(self.relations), self.param)
    rank = self.scorer.evaluation(self.model, self.test)
    self.result = measures(rank)

  def prediction(self, test):
    return self.model.predict(test.indexes)

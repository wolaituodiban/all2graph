import json
from ..meta_graph import MetaEdge
from ..stats import Discrete


class TfidfEdge(MetaEdge):
    TF = 'tf'
    IDF = 'idf'
    """
    记录tf-idf的特殊边类型：
        将一个前置节点作为一个document
        将一个后置节点作为一个term
        据此计算tf-idf值，并储存
    """
    def __init__(self, x, y, num_succs, tf: Discrete, idf: Discrete, **kwargs):
        super().__init__(x, y, num_samples=num_succs, **kwargs)
        assert len(tf.frequency) == len(idf.frequency) == len(set(tf.frequency).union(idf.frequency))
        self.tf = tf
        self.idf = idf

    def to_json(self):
        output = super().to_json()
        output[self.TF] = self.tf.to_json()
        output[self.IDF] = self.idf.to_json()
        return output

    @classmethod
    def from_json(cls, obj):
        if isinstance(obj, str):
            obj = json.loads(obj)
        obj[cls.TF] = Discrete.from_json(obj[cls.TF])
        obj[cls.IDF] = Discrete.from_json(obj[cls.IDF])
        return super().from_json(obj)

    @classmethod
    def from_data(cls, **kwargs):
        raise NotImplementedError

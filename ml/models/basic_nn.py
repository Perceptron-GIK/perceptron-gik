import torch
from torch import nn
from torch.nn import functional
from torch import optim
import matplotlib.pyplot as plt
import pandas as pd
import os 

class Data_stream():
    def __init__(self, file_name: str):
        """_summary_

        Args:
            file_name (str): _description_
        """
        data_pd = pd.read_csv(file_name)
        self.data = torch.tensor(data_pd.to_numpy())
        categories = self._generate_categories()
        self.output_dim = len(categories)
        self.input_dim = self.data.shape

    def _generate_categories(self,):
        """_summary_

        Returns:
            _type_: _description_
        """
        start_l,end_l = ord('a'), ord('z')
        start_num,end_num = ord('0'), ord('9')
        space, enter = ord(' '), ord('\n')
        vec = list(range(start_l,end_l)) + list(range(start_num,end_num)) + [space,] + [enter,]
        return torch.tensor()

    def transform(self):
        """Transform data into different forms (batch, input_dim, output_dim)
        """

# class Models(nn.modules):
#     def __init__(self):
#         pass
#

cwd  = os.getcwd()
file = "input/imu_raw.csv"
dummy = Data_stream(os.path.join(os.path.dirname(cwd),file))


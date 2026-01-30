import pandas as pd 
import os 


class TrainingData:
    def __init__(self, filename: str, data_dir: str):
        self.cur_index = 0 
        self._data_dir = data_dir
        self.file_names = [filename,]
        self.load_data(filename)
        
    def _check_file_exists(self, filename: str) -> str :
        """Checks whether the file exists in the data directory. If it does it returns the file path

        Args:
            filename (str): the filename to check if it exsits 

        Returns:
            str: returns the filepath
        """
        file_path = os.path.join(self._data_dir, filename)
        assert(os.path.isfile(file_path))
        return file_path

    def add_data(self, filename:str):
        self._check_file_exists(filename)
        new_df = pd.read_csv(filename)
        self.df = pd.concat([self.df, new_df], axis=0, ignore_index=True)
        self.filename.append(filename)
        
    def load_data(self, filename: str):
        """_summary_

        Input:
            filename (str): file to be loaded it must be a *.csv file
        """
        self.df = None
        
        # check files exist and if it does get filepath
        file_path = self._check_file_exists(filename)  
        
        self.df = pd.read_csv(file_path)
        
        
        
class Preprocessing:
    def __init__(self, data_dir : str = None,left_file : str = None, right_file : str = None, keyboard_file:str =None ):
        self.data_dir_str =  data_dir
        if right_file is not None : 
            self.right =  TrainingData(right_file, self.data_dir_str)
        if left_file is not None: 
            self.left =  TrainingData(left_file, self.data_dir_str)
        
        assert(keyboard_file is not None)
        self.keyboard =  TrainingData(keyboard_file, self.data_dir_str)
        
        
    def align_time_stamps(self,):
        """_summary_
        """
        
        number_of_chars = self.keyboard.
        self.left_file.df = 
        
        
        
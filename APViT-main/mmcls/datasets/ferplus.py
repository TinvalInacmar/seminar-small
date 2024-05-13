from .builder import DATASETS

from .base_fer_dataset import BaseFerDataset

@DATASETS.register_module()
class FERPlus(BaseFerDataset):
    DATASET_CLASSES = ['Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
    CONVERT_TABLE = (0, 1, 2, 3, 4, 5, 15, 15)
    
    def convert_gt_label(self, i:int):
        """# dataset -> FER_BASE_CLASSES"""
        convert_table = self.CONVERT_TABLE
        #assert sum(convert_table) == sum([i for i in range(8)])
        return convert_table[i]
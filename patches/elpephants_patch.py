import os
import numpy as np
import pandas as pd
from wildlife_datasets.datasets.elpephants import ELPephants as BaseELPephants
from wildlife_datasets.datasets import utils

class PatchedELPephants(BaseELPephants):
    """ELPephants dataset patched to ignore preprocessed images."""

    def create_catalogue(self) -> pd.DataFrame:
        data = utils.find_images(self.root)
        # Remove images created during preprocessing (e.g., segmented datasets)
        exclude_pattern = ['segmented_dataset', 'feature_descriptors_train',
                           'feature_descriptors_test', 'db']
        mask = ~data['path'].str.contains('|'.join(exclude_pattern))
        data = data[mask]

        def safe_extract_identity(x: str):
            try:
                return int(x.strip().split('_')[0])
            except ValueError:
                return np.nan

        df = pd.DataFrame({
            'image_id': utils.create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': data['file'].apply(safe_extract_identity),
            'date': data['file'].apply(self.extract_date),
            'orientation': data['file'].apply(self.extract_orientation),
        })

        df = df[df['identity'].notna()]

        path_txt = utils.find_images(self.root, img_extensions='.txt')
        idx_train = np.where(path_txt['file'] == 'train.txt')[0]
        idx_test = np.where(path_txt['file'] == 'val.txt')[0]
        if len(idx_train) == 1 and len(idx_test) == 1:
            data_train = pd.read_csv(os.path.join(self.root, path_txt['path'].iloc[idx_train[0]], 'train.txt'), header=None, sep='\t')
            data_train = data_train[1].to_numpy()
            data_test = pd.read_csv(os.path.join(self.root, path_txt['path'].iloc[idx_test[0]], 'val.txt'), header=None, sep='\t')
            data_test = data_test[1].to_numpy()
            df['original_split'] = data['file'].apply(lambda x: utils.get_split(x, data_train, data_test))

        return self.finalize_catalogue(df)
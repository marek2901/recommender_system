import os
import urllib
import zipfile

from data_path_helper import datasets_path

class ExtractService(object):

    def __init__(self):
        super(ExtractService, self).__init__()
        self.complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
        self.small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
        self.datasets_path = datasets_path()

        self.complete_dataset_path = os.path.join(self.datasets_path, 'ml-latest.zip')
        self.small_dataset_path = os.path.join(self.datasets_path, 'ml-latest-small.zip')

    def call(self):
        small_f = urllib.urlretrieve(self.small_dataset_url, self.small_dataset_path)
        complete_f = urllib.urlretrieve(self.complete_dataset_url, self.complete_dataset_path)

        for to_extract_path in [self.small_dataset_path, self.complete_dataset_path]:
            with zipfile.ZipFile(to_extract_path, 'r') as z:
                z.extractall(self.datasets_path)


if __name__ == "__main__":
    ExtractService().call()

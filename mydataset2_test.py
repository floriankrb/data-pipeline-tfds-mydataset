"""mydataset dataset."""
import climetlab as cml
import tensorflow_datasets as tfds

import mydataset2


class Mydataset2Test(tfds.testing.DatasetBuilderTestCase):
    DATASET_CLASS = mydataset2.Mydataset2
    SPLITS = {
        "train": 16,  # Number of fake train example
        "test": 6,  # Number of fake test example
    }


if __name__ == "__main__":
    tfds.testing.test_main()

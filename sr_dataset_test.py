"""sr_dataset dataset."""

import tensorflow_datasets as tfds
from sr_dataset import SrDataset


class SrDatasetTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for sr_dataset dataset."""

    # TODO(sr_dataset):
    DATASET_CLASS = SrDataset
    SPLITS = {
        "train": 3,  # Number of fake train examples
        "valid": 1,  # Number of fake validation examples
        "test": 1,  # Number of fake test examples
    }

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == "__main__":
    tfds.testing.test_main()

"""mydataset dataset."""
import climetlab as cml
import tensorflow_datasets as tfds

import mydataset

# ds = cml.load_dataset('s2s-ai-challenge-training-input', date='20200102', parameter='t2m')
# print(ds.to_xarray())


class MydatasetTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for mydataset dataset."""

    # TODO(mydataset):
    DATASET_CLASS = mydataset.Mydataset
    SPLITS = {
        "train": 440,  # Number of fake train example
        "test": 102,  # Number of fake test example
    }

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == "__main__":
    tfds.testing.test_main()

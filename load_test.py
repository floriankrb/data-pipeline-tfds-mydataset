# Load a sample from the new dataset
# NB:
#
# (1) the dataset must have been built 1st - otherwise the import sr_dataset will fail
# (2) download=False means `tfds` should know about the dataset. create a symlink under ~/tensorflow_datasets/sr_dataset
#     that points to the folder where you keep the data
# (3) as_supervised=True will produce only (X_lr, Y) pairs. Not sure how to incorporate the X_hr in that logic.

import sr_dataset
import tensorflow_datasets as tfds


def main():
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        "sr_dataset", split=["train", "valid", "test"], shuffle_files=True, as_supervised=False, with_info=True, download=False
    )

    for batch in ds_train.take(1):
        print(list(batch.keys()))  # ['X_hr', 'X_lr', 'Y']

    for batch in ds_train.take(1):
        print(batch["X_lr"].shape, batch["X_hr"].shape, batch["Y"].shape)  # (1, 16, 16) (3, 128, 128) (1, 128, 128)


if __name__ == "__main__":
    main()

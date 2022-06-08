import os

import pandas as pd
import xarray as xr
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

# TODO(sr_dataset): Markdown description that will appear on the catalog page.
_DESCRIPTION = """Super-resolution dataset. Contains CAMS-regional `pm2.5` analysis data at different resolutions."""

# TODO(sr_dataset): BibTeX citation
_CITATION = """N/A"""

DATA_DIR = "/data/malexe/scratch/TF-DATASETS/raw"

DATE_RANGES = {
    "train": "2014010100_2020110500",
    "valid": "2020110501_2021060108",
    "test": "2021060109_2021103123",
}

REGION_IDS = [17, 18]

LOWRES_VAR = ["pm2p5"]
HIRES_VAR = ["pm2p5"]
CONST_VAR = ["built_frac", "orog_scal", "lsm"]

# NB: the data should be of type float32

# low-res inputs; "name" can be one of ["train", "valid", "test"]
LOWRES_FNAME_TEMPLATE = "cams_eu_aq_16x16_hourly_preproc_{date_range}_region_{reg_id:03d}_{name}.nc"
# hi-res outputs
HIRES_FNAME_TEMPLATE = "cams_eu_aq_128x128_hourly_preproc_{date_range}_region_{reg_id:03d}_{name}.nc"
# constant fields (optional)
CONST_FNAME_TEMPLATE = "const_fields_eu_128x128_preproc_{date_range}_region_{reg_id:03d}_{name}.nc"

# shapes: (nvar, lat, lon)
LOWRES_SHAPE = (1, 16, 16)
HIRES_SHAPE = (1, 128, 128)
CONST_SHAPE = (len(CONST_VAR), 128, 128)


class SrDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for sr_dataset dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(sr_dataset): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    # Hi-res (constant) input features
                    "X_hr": tfds.features.Tensor(shape=CONST_SHAPE, dtype=tf.dtypes.float32, encoding=tfds.features.Encoding.ZLIB),
                    # Low-res input features
                    "X_lr": tfds.features.Tensor(shape=LOWRES_SHAPE, dtype=tf.dtypes.float32, encoding=tfds.features.Encoding.ZLIB),
                    # Hi-res target
                    "Y": tfds.features.Tensor(shape=HIRES_SHAPE, dtype=tf.dtypes.float32, encoding=tfds.features.Encoding.ZLIB),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("X_lr", "Y"),  # Set to `None` to disable
            homepage="ecmwf.int",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        del dl_manager  # not used

        # Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples("train"),
            "valid": self._generate_examples("valid"),
            "test": self._generate_examples("test"),
        }

    def __to_numpy(self, ds: xr.Dataset) -> np.ndarray:
        return np.stack([ds[var].to_numpy() for var in ds.data_vars], axis=0)

    def _generate_examples(self, name):
        """Yields examples."""
        for region_id in REGION_IDS:
            inputname_lr = LOWRES_FNAME_TEMPLATE.format(
                date_range=DATE_RANGES[name],
                reg_id=region_id,
                name=name,
            )

            inputname_hr = HIRES_FNAME_TEMPLATE.format(
                date_range=DATE_RANGES[name],
                reg_id=region_id,
                name=name,
            )

            inputname_const = CONST_FNAME_TEMPLATE.format(
                date_range=DATE_RANGES[name],
                reg_id=region_id,
                name=name,
            )

            # OK to use open_dataset here?
            ds_x_lr = xr.load_dataset(os.path.join(DATA_DIR, name, inputname_lr))
            ds_x_lr = ds_x_lr[LOWRES_VAR]

            ds_y = xr.load_dataset(os.path.join(DATA_DIR, name, inputname_hr))
            ds_y = ds_y[HIRES_VAR]

            ds_x_const = xr.load_dataset(os.path.join(DATA_DIR, name, inputname_const))
            ds_x_const = ds_x_const[CONST_VAR]

            # sanity check
            xr.testing.assert_equal(ds_x_lr.time, ds_y.time)

            for ctime in ds_x_lr.time.values:
                ctime_fmt = pd.to_datetime(str(ctime)).strftime("%Y%m%d%H")
                x_lr = ds_x_lr.sel(time=ctime, drop=True)
                y = ds_y.sel(time=ctime, drop=True)

                yield f"key-{name}-reg{region_id:03d}-{ctime_fmt}", {
                    "X_hr": self.__to_numpy(ds_x_const),
                    "X_lr": self.__to_numpy(x_lr),
                    "Y": self.__to_numpy(y),
                }

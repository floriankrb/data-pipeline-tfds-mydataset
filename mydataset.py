"""mydataset dataset."""
import climetlab as cml
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

SUBSAMPLE, SHAPE = False, (46, 121, 240)
SUBSAMPLE, SHAPE = 20, (46, 121 // 20 + 1, 240 // 20)


# ds = cml.load_dataset(
#     "s2s-ai-challenge-training-input",
#     date="20200102",
#     parameter="t2m",
# )


# TODO(mydataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(mydataset): BibTeX citation
_CITATION = """cite"""


class Mydataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mydataset dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "t2m": tfds.features.Tensor(shape=SHAPE, dtype=tf.dtypes.float32),
                    "obs": tfds.features.Tensor(shape=SHAPE, dtype=tf.dtypes.float32),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("t2m", "obs"),  # Set to `None` to disable
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(mydataset): Downloads the data and defines the splits
        # path = dl_manager.download_and_extract('https://todo-data-url')

        return {
            "train": self._generate_examples("train"),
            "test": self._generate_examples("test"),
        }

    def _generate_examples(self, name):
        """Yields examples."""
        inputname = dict(
            train="s2s-ai-challenge-training-input",
            test="s2s-ai-challenge-test-input",
        )[name]

        dates = ["20200102", "20200109"]
        for date in dates:
            xds = cml.load_dataset(inputname, date=date, parameter="t2m")
            xds = xds.to_xarray()

            outputname = dict(
                train="s2s-ai-challenge-training-output-reference",
                test="s2s-ai-challenge-test-output-reference",
            )[name]
            yds = cml.load_dataset(outputname, date=date, parameter="t2m")
            yds = yds.to_xarray()
            if float(yds.lead_time[0]) == 0:
                # remote first lead_time if it is zero (t2m for ecmwf)
                yds = yds.sel(lead_time=yds.lead_time[1:])
            if SUBSAMPLE:
                xds = xds.sel(
                    latitude=slice(None, None, SUBSAMPLE),
                    longitude=slice(None, None, SUBSAMPLE),
                )
                yds = yds.sel(
                    latitude=slice(None, None, SUBSAMPLE),
                    longitude=slice(None, None, SUBSAMPLE),
                )

            assert np.all(yds.forecast_time.to_numpy() == xds.forecast_time.to_numpy())
            for time in xds.forecast_time:
                yda = yds["t2m"]
                yda = yda.sel(forecast_time=time)
                for i in xds.realization:
                    xda = xds["t2m"]
                    xda = xda.sel(forecast_time=time)
                    xda = xda.sel(realization=i)

                    yield f"key-{name}-{date}-{i}-{time}", {
                        "t2m": xda.to_numpy(),
                        "obs": yda.to_numpy(),
                    }

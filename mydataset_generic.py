"""mydataset dataset."""
import climetlab as cml
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

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


class MydatasetGeneric(tfds.core.GeneratorBasedBuilder):
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
                    "t2m": tfds.features.Tensor(
                        shape=self.SHAPE, dtype=tf.dtypes.float32
                    ),
                    "obs": tfds.features.Tensor(
                        shape=self.SHAPE, dtype=tf.dtypes.float32
                    ),
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
        outputname = dict(
            train="s2s-ai-challenge-training-output-reference",
            test="s2s-ai-challenge-test-output-reference",
        )[name]

        fctime_len = dict(train=20, test=1)[name]
        realization_len = dict(train=11, test=51)[name]
        if self.dev:
            fctime_len = dict(train=2, test=1)[name]
            realization_len = dict(train=4, test=3)[name]

        DATES = ["20200102", "20200109"]

        zipped = [
            (d, time, r)
            for d in DATES
            for time in range(fctime_len)
            for r in range(realization_len)
        ]

        def _process_example(args):
            date, time, realization = args

            xds = cml.load_dataset(inputname, date=date, parameter="t2m")
            xds = xds.to_xarray()

            yds = cml.load_dataset(outputname, date=date, parameter="t2m")
            yds = yds.to_xarray()

            if float(yds.lead_time[0]) == 0:
                # remote first lead_time if it is zero (t2m for ecmwf)
                yds = yds.sel(lead_time=yds.lead_time[1:])

            if self.dev:
                xds = xds.sel(
                    latitude=slice(None, None, self.SUBSAMPLE),
                    longitude=slice(None, None, self.SUBSAMPLE),
                )
                yds = yds.sel(
                    latitude=slice(None, None, self.SUBSAMPLE),
                    longitude=slice(None, None, self.SUBSAMPLE),
                )

            if not self.dev:
                assert len(xds.forecast_time) == fctime_len, xds.forecast_time
                assert len(xds.realization) == realization_len, xds.realization
                assert len(yds.forecast_time) == fctime_len, yds.forecast_time

            assert np.all(yds.forecast_time.to_numpy() == xds.forecast_time.to_numpy())

            yda = yds["t2m"]
            yda = yda.isel(forecast_time=time)

            xda = xds["t2m"]
            xda = xda.isel(forecast_time=time)
            xda = xda.isel(realization=realization)

            return f"key-{name}-{date}-{realization}-{time}", {
                "t2m": xda.to_numpy(),
                "obs": yda.to_numpy(),
            }

        beam = tfds.core.lazy_imports.apache_beam
        return beam.Create(zipped) | beam.Map(_process_example)

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

        beam = tfds.core.lazy_imports.apache_beam
        return beam.Create(zipped) | beam.Map(_process_example)

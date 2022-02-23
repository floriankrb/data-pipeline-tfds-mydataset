import logging
import os
import shutil
import sys
from datetime import datetime

import apache_beam as beam
import climetlab as cml
import numpy as np
import tensorflow as tf
import xarray as xr

tf.autograph.set_verbosity(0)

logging.basicConfig(level="WARN")

# SHAPE = (46, 121, 240)
SUBSAMPLE, IS_DEV = False, False

# SHAPE = (46, 121 // 20 + 1, 240 // 20)
# SUBSAMPLE, IS_DEV = 20, True


def _array_feature(value, min_value=None, max_value=None):
    value = np.array([1.0, 2.0])
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    LOG = logging.getLogger(__name__)
    if isinstance(value, type(tf.constant(0))):  # if value is tensor
        value = value.numpy()  # get value of tensor

    """Wrapper for inserting ndarray float features into Example proto."""
    value = np.nan_to_num(value.flatten())  # nan, -inf, +inf to numbers
    if min_value is not None and max_value is not None:
        value = np.clip(value, min_value, max_value)  # clip to valid
    logging.info("Range of image values {} to {}".format(np.min(value), np.max(value)))
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def generate_zipped(name):
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
    if IS_DEV:
        fctime_len = dict(train=2, test=1)[name]
        realization_len = dict(train=4, test=3)[name]

    DATES = ["20200102", "20200109"]

    # one worker loops on all realizations
    # realization_lists = [range(realization_len)]

    # one worker per realization
    realization_lists = [[r] for r in range(realization_len)]

    # the actual use case is where we want all realizations in a single example
    # realization_lists = [[None]]

    zipped = [
        (
            inputname,
            outputname,
            d,
            time,
            fctime_len,
            realizations,
            realization_len,
        )
        for d in DATES
        for time in range(fctime_len)
        for realizations in realization_lists
    ]
    zipped = [(i, len(zipped), *args) for i, args in enumerate(zipped)]
    # np.random.shuffle(zipped)
    return zipped


def process_example(args):
    try:
        (
            i,
            i_total,
            inputname,
            outputname,
            date,
            time,
            fctime_len,
            realizations,
            realization_len,
        ) = args

        xds = cml.load_dataset(inputname, date=date, parameter="t2m")
        xds = xds.to_xarray()

        yds = cml.load_dataset(outputname, date=date, parameter="t2m")
        yds = yds.to_xarray()

        if float(yds.lead_time[0]) == 0:
            # remote first lead_time if it is zero (t2m for ecmwf)
            yds = yds.sel(lead_time=yds.lead_time[1:])

        if IS_DEV:
            xds = xds.sel(
                latitude=slice(None, None, SUBSAMPLE),
                longitude=slice(None, None, SUBSAMPLE),
            )
            yds = yds.sel(
                latitude=slice(None, None, SUBSAMPLE),
                longitude=slice(None, None, SUBSAMPLE),
            )

        print("asserting")

        if not IS_DEV:
            assert len(xds.forecast_time) == fctime_len, xds.forecast_time
            assert len(xds.realization) == realization_len, xds.realization
            assert len(yds.forecast_time) == fctime_len, yds.forecast_time

        assert np.all(yds.forecast_time.values == xds.forecast_time.values)
        print(f"assertions ok")

        yda = yds["t2m"]
        yda = yda.isel(forecast_time=time)

        xda = xds["t2m"]
        xda = xda.isel(forecast_time=time)

        for j, realization in enumerate(realizations):
            if realization is not None:
                xda_ = xda.isel(realization=realization)
            else:
                xda_ = xda

            def to_feat(arr):
                value = arr
                value = value.astype("float")
                value = np.nan_to_num(value.flatten())  # nan, -inf, +inf to numbers
                feat = tf.train.Feature(float_list=tf.train.FloatList(value=value))
                return feat

            tfexample = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "t2m": to_feat(xda_.values),
                        "obs": to_feat(yda.values),
                    }
                )
            )
            to_yield = tfexample.SerializeToString()
            print(f"Writing example {i+1}/{i_total}:{j} OK. {datetime.now()}")
            yield to_yield
    except:
        e = sys.exc_info()[0]
        raise (e)
        print(e)
        logging.error(e)


def run_job(flags, options, name, outdir):

    # start the pipeline
    opts = beam.pipeline.PipelineOptions(flags=flags, **options)
    with beam.Pipeline(options["runner"], options=opts) as p:
        # create examples
        examples = (
            p
            | "generate_example_args" >> beam.Create(generate_zipped(name))
            | "create_tfrecord" >> beam.FlatMap(lambda x: process_example(x))
        )

        # write out tfrecords
        _ = examples | "write_tfr" >> beam.io.tfrecordio.WriteToTFRecord(
            os.path.join(outdir, "tfrecord")
        )


def main(name):
    outdir = f"outdir/{name}"
    print("Launching local job ... hang on")
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir)

    options = dict()
    flags = []
    # ----- Direct runner
    # options["runner"] = "DirectRunner"
    # options['direct_num_workers']=0

    # ----- Dataflow
    # options['runner'] = 'DataflowRunner'

    # ----- Spark
    options["runner"] = "PortableRunner"
    flags = [
        "--job_endpoint=192.168.1.250:7077", # sparkmaster
        #"--environment_type=LOOPBACK",
        ]

    run_job(flags, options, name, outdir)


if __name__ == "__main__":
    main("test")
    # main("train")

# tfds experiments

Assuming all packages are installed.
including
```
pip install apache-beam
pip install tensorflow_datasets
pip install climetlab climetlab-s2s-ai-challenge
```


Run this with:
```
tfds build dataset2.py  # small test dataset: works
tfds build dataset.py  # larger dataset, need apache beam and crash with bad_aloc or out of memory


python ./mydataset2_test.py # test pass
python ./mydataset_test.py  #   std::bad_alloc  out of memory

tfds build --max_examples_per_split 2 mydataset.py  # does not help
```

### License
[Apache License 2.0](LICENSE) In applying this licence, ECMWF does not waive the privileges and immunities 
granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.

 

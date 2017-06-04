# fast-image-retrieval
A lightweight framework using binary hash codes and deep learning for fast image retrieval.

## Configuration
To run the examples, you need to create a `config.cfg` file under the root folder of this project. An example of `config.cfg` looks like:

```
[examples]
shoes7k_pos_path: /path/to/datasets/shoes7k/classification
shoes7k_neg_path: /path/to/datasets/shoes7k/classificationNeg
```

## Run Examples

To run example on shoes7k dataset, firstly, you need to convert shoes7k dataset to LMDB dataset.

```
cd fast-image-retrieval/
python ./examples/shoes7k/convert_shoes7k_data.py
```

Then, you need to train the CNN model.

```
./examples/shoes7k/train.sh
```

Next, you can retrieve similar image `target.jpg` using

```
./examples/shoes7k/retrieve.sh target.jpg
```

Note that the first retrival procedure might be very slow because the program reads all shoes7k images and generate feature matrix. The later retrivals can be very fast.

# Team NoreGraph - CSED/AIGS538 Deep Learning

Uploaded 3 results.

```
$ python train_u2gnn_sup.py --model_name KAGGLE --load_epoch 33 --test_only

Namespace(batch_size=16, dropout=0.5, hidden_size=128, learning_rate=0.0001, load_epoch=33, model_name='KAGGLE', num_epochs=50, num_hidden_layers=1, num_neighbors=16, num_timesteps=1, test_only=True)
Loading data...
# data: 5000 | # classes: 3 | # max node tag: 367
Loading data finished.
node features dimension: 370
Using /home/abc/NoreGraph/runs/U2GNNsup/KAGGLE
state loaded from epoch 33 - test_acc: 72.17%
```

### uGCN

```sh
$ python train_ugcn.py
usage: train_ugcn.py [-h] [--model_name MODEL_NAME]
                     [--learning_rate LEARNING_RATE]
                     [--batch_size BATCH_SIZE]
                     [--num_epochs NUM_EPOCHS]
                     [--num_sampled NUM_SAMPLED]
                     [--hidden_size HIDDEN_SIZE]
                     [--num_conv_layers NUM_CONV_LAYERS]
                     [--dropout DROPOUT]
                     [--no_soft_placement]
                     [--log_device_placement]
                     [--load_epoch LOAD_EPOCH]
                     [--test_only]

optional arguments:
  -h, --help            show this help message and exit
  --model_name          Output directory name (default: KAGGLE)
  --learning_rate       Learning rate (default: 0.0001)
  --batch_size          Batch size (default: 128)
  --num_epochs          Number of training epochs (default: 50)
  --num_sampled         Sampled softmax length to embedding (default: 256)
  --hidden_size         The hidden layer size (default: 128)
  --num_conv_layers     Number of stacked graph convolution layers (default: 2)
  --dropout             Dropout rate (default: 0.5)
  --no_soft_placement   Disallow device soft device placement (default: False)
  --log_device_placement    Log placement of ops on devices (default: False)
  --load_epoch          Load previous state if set (default: 0)
  --test_only           Print test result and exit (default: False)
```

### unsupU2GNN

```sh
$ python train_u2gnn_unsup.py -h
usage: train_u2gnn_unsup.py [-h] [--model_name MODEL_NAME]
                            [--learning_rate LEARNING_RATE]
                            [--batch_size BATCH_SIZE]
                            [--num_epochs NUM_EPOCHS]
                            [--num_sampled NUM_SAMPLED]
                            [--hidden_size HIDDEN_SIZE]
                            [--num_hidden_layers NUM_HIDDEN_LAYERS]
                            [--num_timesteps NUM_TIMESTEPS]
                            [--num_neighbors NUM_NEIGHBORS]
                            [--dropout DROPOUT]
                            [--load_epoch LOAD_EPOCH]
                            [--test_only]

optional arguments:
  -h, --help            show this help message and exit
  --model_name          Output directory name (default: KAGGLE)
  --learning_rate       Learning rate (default: 0.0001)
  --batch_size          Batch size (default: 16)
  --num_epochs          Number of training epochs (default: 50)
  --num_sampled         Sampled softmax length to embedding (default: 256)
  --hidden_size         The hidden size for the feedforward layer (default: 128)
  --num_hidden_layers   Number of hidden layers in the encoder (default: 1)
  --num_timesteps       Timestep T ~ Number of self-attention layers within each U2GNN layer (default: 1)
  --num_neighbors       Number of neighbors for the input of the encoder (default: 16)
  --dropout             Dropout rate (default: 0.5)
  --load_epoch          Load previous state if set (default: 0)
  --test_only           Print test result and exit (default: False)
```

### supU2GNN

```sh
$ python train_u2gnn_sup.py -h
usage: train_u2gnn_sup.py [-h] [--model_name MODEL_NAME]
                          [--learning_rate LEARNING_RATE]
                          [--batch_size BATCH_SIZE]
                          [--num_epochs NUM_EPOCHS]
                          [--hidden_size HIDDEN_SIZE]
                          [--num_hidden_layers NUM_HIDDEN_LAYERS]
                          [--num_timesteps NUM_TIMESTEPS]
                          [--num_neighbors NUM_NEIGHBORS]
                          [--dropout DROPOUT]
                          [--load_epoch LOAD_EPOCH]
                          [--test_only]

optional arguments:
  -h, --help            show this help message and exit
  --model_name          Output directory name (default: KAGGLE)
  --learning_rate       Learning rate (default: 0.0001)
  --batch_size          Batch size (default: 16)
  --num_epochs          Number of training epochs (default: 50)
  --hidden_size         The hidden size for the feedforward layer (default: 128)
  --num_hidden_layers   Number of hidden layers in the encoder (default: 1)
  --num_timesteps       Timestep T ~ Number of self-attention layers within each U2GNN layer (default: 1)
  --num_neighbors       Number of neighbors for the input of the encoder (default: 16)
  --dropout DROPOUT     Dropout rate (default: 0.5)
  --load_epoch          Load previous state if set (default: 0)
  --test_only           Print test result and exit (default: False)
```

### test

Assume that the dataset is COLLAB.

```sh
python test_accuracy.py -h
usage: test_accuracy.py [-h] filepath

positional arguments:
  filepath    Sample CSV to test

optional arguments:
  -h, --help  show this help message and exit
```

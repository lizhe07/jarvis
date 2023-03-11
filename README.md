# jarvis
`jarvis` is designed to be a butler that helps model training.

Run the `pip` install commands at the repository folder to install the package.
```bash
pip install .
```

## Manager
`jarvis` uses `Manager` class to manage trainings with different configurations. Each training is
assigned with a unique key (default length 8), and its configuration and checkpoint along with some
auxiliary information are stored in separate archive directories.

### Initialization
A `Manager` object is initialized with:
- `store_dir`, directory for storing training data
- `defaults` (Optional), a dictionary or a yaml file name containing one that provides default
values of a training configuration.
```python
store_dir = 'store'
defaults = 'defaults.yaml'
manager = Manager(store_dir, defaults)
```
Four sub-directories will be created in `'store'` if not existing.
- `'configs'`: for training configuraitons, mutually exclusive from each other.
- `'stats'`: for training status, including the number of epochs trained.
- `'ckpts'`: for checkpoints, typically stores the model state dictionary.
- `'previews'`: for previews of each training, useful to compare trained models without loading
them.

### Train a model
A few methods need to be overridden or implemented before the `manager` can work properly, the
requirements for subclassing the `Manager` will be detailed in
[this section](###pipeline-and-subclassing).

After a `Manager` class has been properly subclassed, we can train one model as below.
```python
manager.process(config, num_epochs=num_epochs)
```
`config` is a dictionary of training specification, such as the dataset to train on, the
architecture of model, learning rate of the optimizer and etc. For example,
`config = {'arch': 'ResNet18'}`.

`num_epochs` is an integer for how many epochs to train.

### Train a batch of models
The `manager` can train models for a batch of configurations in random (or designed) order. This
feature is designed for cluster use, so that many machines can work on a same set of tasks in
parallel.
```python
manager.batch(configs, num_epochs=num_epochs, count=count)
```
`count` is the number of models to be trained. More details and additional supported keyword
arguments can be found in the doc string.

If the configurations are defined through a grid of configuration values, the `manager` can sweep
over the combinations like below.
```python
manager.sweep(choices)
```
`choices` is a dictionary with the same structure of a valid `config`, except its leaf values are
lists of corresponding values of a `config`. For example,
`choices = {'arch': ['ResNet18', 'ResNet50'], 'lr': [0.01, 0.001]}`.

### Pipeline and subclassing
Here is a brief description of the pipeline of `manager.process`, along with the list of methods
that can be overridden or need to be implemented in subclass of `Manager`.

To train a model, the `manager` first prepares a `Config` object by filling default values from
`manager.defaults` to a partially specified configuration dictionary.
```python
config = manager.get_config(config)
```
`get_config` method can be overridden to include additional checks if necessary.

Next the `manager` sets up some attributes and serves as a working space via
`manager.setup(config)`. `manager` next tries to load checkpoint via `manager.load_ckpt()`. If not
successful, it initializes a checkpoint via `manager.init_ckpt()`.

`manager` then iteratively train the model for the desired number of epochs by calling
`manager.train()`. `manager` also evaluates the model (usually on a validation dataset) by calling
`manager.eval()` every a few epochs specified by `manager.eval_interval`. Finally, at every
`manager.save_interval` epoch or at the end of training, `manager` saved the current checkpoint via
`manager.save_ckpt()`.

To summarize, the user needs to implement
- `manager.train()`
- `manager.eval()`

and override
- `manager.get_config()` (optional)
- `manager.setup(config)`
- `manager.init_ckpt()`
- `manager.save_ckpt()`
- `manager.load_ckpt()`

in the subclass of `Manager`. More details can be found in doc strings of each methods in
`jarvis.manager.Manager`.

### Fetch a trained model
Given the key of a trained model as `key`, we can fetch the model by loading the checkpoint and
return the corresponding attribute of `manager`.
```python
config = manager.configs[key]
manager.setup(config)
manager.load_ckpt()
model = manager.model
```

# CoCoa: Conditional Compound Generation using Continous Normalizing Flow

## What is the Purpose of CoCoa

CoCoa is a machine learning method for generation of novel chemical compounds that exhibit specifics values of certian chemical properties. The major benifit of CoCoa is that it allows for direct control over property values of generated compounds, instead of tring to minimize or maximize the value. Additionally, CoCoa is desiged to be flexible and can work with any simple numerical chemical property (complex properties are not currently supported) and can even generate new compounds based on several properties simultaneously.

To utalize CoCoa for user specific purposes, the user must provide a dataset of chemical compounds (repersented as [SMILES](https://daylight.com/dayhtml/doc/theory/theory.smiles.html) or [SELFIES](https://github.com/aspuru-guzik-group/selfies)) and each compounds' associated chemical property value for all properties that may be used in generation. This will be described in detail later in this guide. For optimal performance, the dataset property range should encompass the property values which you plan to sample (i.e. sampling outside the dataset property range can lead to decreased performance). 

## CoCoa Manuscript

Please read our manuscript at [TODO: PLACE ARCHIVE LINK HERE](https://github.com/chopralab) for additional information on CoCoa.

## How to Obtain CoCoa

Cloning the GitHub repository is currently the only supported way to acquire CoCoa. You can clone the GitHub repository with the following command (TODO: update this with the version on ChopraLab GitHub):

```
git clone https://github.com/chopralab
```

TODO: Are we adding any additional way to get the source code (build, pip, conda, etc?)
****Probably need to add a requirements.txt****

## Scripts for Training and Sampling CoCoa Models

### `main.py`

This script is used to train an Encoder/Decoder and CNF (Continous Normalizing Flow) model based on a user provided training dataset. The command line options for the script are described below:

```
--data-file: The path to the data file containing the training dataset (Type: String, Required)
--log-root: Directory to log training metrics and save checkpoint models (Type: String, Default:'~/Experiments')

--batch-size: Batch size for model training (Type: Integer, Default: 256)
--epoch: Number of training epochs (Type: Integer, Default: 100)
--num-workers: Number of workers for model training (Type: Integer, Default: 0)
--lr: Learning rate for weight updates (Type: Float, Default: 1e-3)
--weight-decay: Weight decay after weight updates (Type: Float, Default: 0)

--input-dims: Dimensionality of VAE encoder input (Type: list, Default: [Length of SELFIES vocabulary,3])
--latent-dim: Dimensionality of the VAE latent space (Type: Integer, Default: 256)

--cnf-hidden-dims: Hidden Dimensionality of the CNF model (Type: List, Default: [256,256,256,256])
--cnf-train-context: Enabling Conditional CNF
--cnf-T: End time hyperparameter for CNF ODE integral
--cnf-train-T: Enable learning CNF ODE integral end time 

--alpha: Hyperparameter to scale entropy and prior loss (Type: Float, Default: 1e-3)

--ode-solver: ODE solver from torchdiffeq
--ode-atol: ODE solver error tolerance
--ode-rtol: ODE solver error tolerance 
--ode-use-adjoint: Use adjoint method for backprop through ODE solver

--decoder-num-layers: Number of layers in the Decoder (Type: Integer, Default: 4)

--pretrained: Path for pretrained model (Type: String, Default: '')
--cuda: Flag for cuda usage for GPU acceleration (True if Flagged)
--track: Log trainng run metrics and save checkpoint models (True if Flagged)
--comment: Comment string to keep track of experiment

--seed: Seeding for random number generation (Type: Integer, Default: 0)
--global-rank: Global process rank for current process (Type: Integer, No Default)
--local--rank: Local process rank for current process (Type: Integer, No Default)
```

### `eval.py`

This script is used to evaluate a trained model by calculating statistics regarding the chemical property values of the sampled compounds. Currently it is only supported for chemical properties logP, TPSA, and SA Score:

```
--model: The path to the trained model to evaluate (Type: String, Required)
--data-file: The path to the dataset that was used to train the model (Type: String, Required)
--datastat-file: The path to the statistics file corresponding to the dataset (Type: String, Required)
--num-mols: The number of molecules to generate and evaluate (Type: Integer, Default: 1000)
--batch-size: The batch size used for sampling (Type: Integer, Default: 32)

--token-dim: The dimensionality of the SELFIES token alphabet used (Type: Integer, Default: 139)
--latent-dim: The dimensionality of the VAE latent space of the trained model (Type: Integer, Default: 256)
--cnf-dims: The dimensionality of the CNF latent space of the trained model (Type: List, Default: [256,256,256,256])

--logp: Flag for use of logP as a property for evaluation sampling (True if Flagged)
--tpsa: Flag for use of TPSA as a property for evaluation sampling (True if Flagged)
--sascore: Flag for use of SA Score as a property for evaluation sampling (True if Flagged)

--logp-dim: Sampling dimension of logP if flagged (Type: Integer, Default: 0, Should be 0, 1, or 2)
--tpsa-dim: Sampling dimension of TPSA if flagged (Type: Integer, Default: 1, Should be 0, 1, or 2)
--sascore-dim: Sampling dimension of SA Score if flagged (Type: Integer, Default: 2, Should be 0, 1, or 2)

--eval-name: Name for evaluate file (Type: String, Default: 'eval')

--nproc: Number of processes to run (Type: Integer, Default: 4)
--cuda: Flag for cuda GPU acceleration (True if Flagged)
```

## Usage Flowchart for CoCoa

### Step 1 - Clone Repository
Clone the GitHub Repository as mentioned in the "How to I Obtain CoCoa" Section

### Step 2 - Conda Enviroment
Provided in the GitHub Repositroy is the `env.yml` file. This file can be used in association with the Anaconda package manager to quickly set up an enviroment containing all packages required to run CoCoa. Create the Conda enviroment with the following command:

```
conda env create --name CoCoa --file env.yml
```

If you wish to update a pre-existing Conda enviroment with the packages, run the following command replacing `[ENV_NAME]` with the name of the pre-existing enviroment:

```
conda env update --name [ENV_NAME] --file env.yml
```

### Step 3 - Training the Model

### Step 4 - Generating New Compounds

## Other Scripts

### Main directory

#### `env.yml`

This is the YAML file that is used by the Anaconda package manager to reconstruct a Conda enviroment that contains all of the required dependencies to run CoCoa. See Step 2 of the CoCoa Usage Flowchart to see how to create a Conda enviroment with all of the dependencies.

#### `eval_original.py`

This is the original evaluation script. It does not contain code for writing out the full statistics for the evaluation run. The `eval.py` script has this code and was used for recording of the statistics for the manuscript.

### `data` subdirectory

This subdirectory containts scripts involved in data processing

#### `homolumo.py`

This is the preprocessing script for SELFIES and HOMO/LUMO data. Reads in a dataframe and constructs tensors with this information.

#### `selfies.py`

This script defines the SELFIES vocabulary (alphabet) that is used to tokenize molecular data, converts SMILES strings to SELFIES tokens, and performs positional embedding on the tokens.

### `figures` subdirectory

This subdirectory contains figures that are used in the Paper and README file.

### `models` subdirectory

This subdirectory contains the ML models that are used in CoCoa.

#### `decoder.py`

This is the script for the context based decoder.

#### `encoder.py`

This is the script for the variational autoencoder

#### `network.py`

This file contains the class for the CVAEF (Conditional Variational AutoEncoder Flow). 

### `model/cnf` subdirectory

This subdirectory contains files and scripts related to the CNF (Continous Normalizing Flow) model

#### `cnf.py`

This file contains the class for the CNF (Continous Normalizing Flow) model

#### `layers.py`

This file contains classes for neural network layer functionality.

#### `normalization.py`

This file contains classes for TODO: I don't really know how this is used ...

#### `ode.py`

This file contaions the classes related to the ODE (Ordinary Differential Equation) solver used to determine the invertible function used in the CNF transformation.

### `notebooks` subdirectory

This subdirectory contains Jupyter notebooks that were used for model testing, sampled compound visualization, and latent space visualization.

### `utils` subdirectory

This subdirectory contains utility scirpts that assist with model training and sampling.

#### `logger.py`

This script is used for creating and saving experiment log files.

#### `process_data.py`

This is the script for statistical analysis calculation of sampled compounds. It calculates the mean and standard deviation of the chemical property values of sampled compounds for each expected property value.

#### `radam.py`

This script provides an implementation of the RAdam optimizer. Taken from https://github.com/LiyuanLucasLiu/RAdam

#### `trainer.py`

This file contains a class for training CoCoa models. The class defines runtime options and model hyperparameters and includes methods for VAE and model training and validation. There are also methods for saving a model (weights) for future use.

## Citation

TODO: Put archive citation here (update to final citation when published)

## Copyright

TODO:  Add copyright and licensing here if needed

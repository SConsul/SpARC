# SpARC: Sparse Activation Regularization for Consistency
The repository is part of a course project for Stanford CS 224N (Winter 2022), taught by [Prof. Chris Manning](https://nlp.stanford.edu/~manning/).


## Installation

### Install the required packages
Use pip to install the following packages:
```bash
pip install -U adapter-transformers
pip install sentencepiece
```
To vizualize the model activations in `notebooks/Visualizer.ipynb`, requires the additional installation of the [BertViz](https://github.com/jessevig/bertviz) package.

### Clone the repository
Clone this repository using:

```bash
git clone https://github.com/SConsul/SpARC.git
```
## Usage

### Data Preprocessing
The BeliefBank dataset is available at https://allenai.org/data/beliefbank

To parse the relation graphs into question-answer pairs, as well as generating the train, val and test splits, run:
```bash
python src/utils/preprocess.py
```
This will generate json files with the required question-answer splits, qa_train, qa_val and qa_test. A separate qa_consistency to test the consistency of the model.

### Finetuning
To finetune the MACAW-large transformer on the BeliefBank dataset, use the following command:
```bash
python src/train.py 
```
**Required arguments:**
| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
|`--train_path`| None | json file containing the training data
|`--model_path`| None| Location to save model weights.

**Optional arguments:**
| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
|`--max_epochs`| 10| number of epochs of finetuning.
|`--batch_size`| 64| batch size used, can be varied depending on the available GPU memory.
|`--lr`| 3e-4| Initial learning rate for Adam Optimizer.
|`--lr_decay`| False| Boolean flag if learning rate should decay.
|`--weight_decay`| 0.1| L2 Regularization penalty on model weights.
|`--num_workers`| 4| Number of workers, can be varied depending on the available CPU memory.
|`--l1_reg`| None| A float value indicating the penalty of th L1 norm of activations.
|`--freeze_backbone`| False| Used to only finetune on the final linear layer of the model.
|`--adapter`| True| Set to true to enable adapter layers for finetuning.
|`--layer_names`| None| List of layers whose activations are to be regularized.
|`--sim`| None| The float hyperparameter weighing the similarity loss.
|`--ce_loss`| 1.0| The float hyperparameter weighing the standard cross-entropy loss.
|`--token_type`| None| Set to 'answer', 'question', 'eos', 'link' .
|`--sim_type`| None| Set to 'angle' if similarity of activations are to be measured using the angles instead of their dot-products.

### Evaluation
Get the inference of a saved model by running:
```bash
python src/inference.py
```
**Required arguments:**
| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
|`--in_path`|./beliefbank-data-sep2021/qa_test.json| choice of dataset split to do evaluate on. 
|`--out_path`| None| Path to save the inferences. 
|`--batch_size`| 512 | batch size used, can be varied depending on the available GPU memory.
|`--model_path`| None| path of saved model weights
|`--adapter`| None| Add flag if the network was finetuned using adapters


To obtain a model's accuracy, collect its inferences on the val/test set and run:
```bash
python src/utils/accuracy.py --results_path <path to inferences on val/test set>
```

To obtain the model's consistency, collect its inferences on qa_consistency and run:
```bash
python src/utils/consistency_v2.py --results_path <path to inferences on qa_consistency>
```

#
Team Members: Julia Xu ([**@jwxu**](https://github.com/jwxu)), Samar Khanna ([**@Dieblitzen**](https://github.com/Dieblitzen)), Sarthak Consul ([**@SConsul**](https://github.com/SConsul))
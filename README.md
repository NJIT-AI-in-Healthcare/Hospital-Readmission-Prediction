# Hospital-Readmission-Prediction
## About The Project
Codes for CHSLM model to predict hospital readmission. The model captures the semantic compositions and hierarchical impacts.

### Dataset
1. To access the MIMIC3 dataset, please go to https://physionet.org/content/mimiciii/1.4/
2. Process data([get_concept_scipy.py](get_concept_scipy.py))


### Run
1. model training:
   ```
   python run.py --output_dir PATH_TO_OUTPUT_RESULT --dropout 0.3 --learning_rate 2e-5 --train_batch_size 8 --eval_batch_size 8 --num_classes 1 --gradient_accumulation_steps 1 --num_train_epochs 2 --logging_steps 500 --dataset_folder PATH_TO_DATA --multi_hop
   ```
3. model evaluation:
   ```
   python run.py --output_dir PATH_TO_OUTPUT_RESULT --dropout 0.3 --learning_rate 2e-5 --train_batch_size 8 --eval_batch_size 8 --num_classes 1 --gradient_accumulation_steps 1 --num_train_epochs 2 --logging_steps 500 --dataset_folder PATH_TO_DATA --load_model --load_classification_path PATH_TO_MODEL --load_graph --graph_dir PATH_TO_SAVED_GRAPH 
   ```

### Models
The trained models can be downloaded here: https://drive.google.com/drive/folders/1Cha8ChjGYHa-eUNTEMEBu_39jbSvbxZH?usp=drive_link

### Built With
* [pytorch](https://pytorch.org/)
* [pyg](https://pyg.org/)
* [transformers](https://huggingface.co/transformers/v4.7.0/installation.html)
* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [nltk](https://www.nltk.org/)
* [scispacy](https://allenai.github.io/scispacy/)

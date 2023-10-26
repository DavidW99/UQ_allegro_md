## How to generate xyz file including the features

Use data_config.yaml to setup the data path (let it be your train, validation, testing data) for generation of the features. Then, we call the following command to generate the features. 

```bash 
nequip-evaluate --train-dir nequip-evaluate --train-dir <train_model_dir> --dataset-config data_config.yaml --batch-size 1 --output <output.xyz> --output-fields edge_features,edge_index,edge_energy
```
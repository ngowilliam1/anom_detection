# Anomaly Detection

## Running the code
Note:
Code uses: scikit-learn:  0.20.1 (for Isolation forest)

Code uses: Tensorflow: 1.14.0

Code uses (for GPU): cudatoolkit: 10.1.168

Code uses (for GPU): cudnn:  7.5.1_10.1

### Generating the preprocessed data
First, ensure datasets are in ../data from the cloned repo
```
# This will generate train/validation/test preprocessed set for desired percentages and dataset
# Generate preprocessed data for 10 different anomaly percentages for kdd
python 00_generate_data.py kdd
# Replace kdd with the desired dataset (ex: credit, mammography)
python 00_generate_data.py mammography
# If instead of needing 10 different anomaly percentages need only say 0.02 and 0.05, use -p argument followed by desired percentage
python 00_generate_data.py kdd -p 0.02 0.05
# If preprocessed needs to be standardized instead of min-max scaling (by default it is min-max), use -scale argument:
python 00_generate_data.py kdd -scale standard
```

### Training/Evaluating

Need to generate preprocessed data before training/evaluating

View source code to select hyperparameters of each model
```
# Train and evaluate kdd (on validation set) Isolation forest model
python 01_evaluate.py kdd IF
# Replace kdd with the desired dataset (ex: credit, mammography)
python 01_evaluate.py credit IF
# Replace Isolation forest model with the desired model (ex: AE, GAN, PID)
python 01_evaluate.py kdd PID
# For AE and GAN GPU usage is supported (disabled by default), use -g argument to select desired gpu
python 01_evaluate.py kdd AE -g 0
# For GAN, IF, PID saving fitted model is currently supported (disabled by default), use -save argument
python 01_evaluate.py kdd GAN -g 0 -save True
# If want to train/evaluate on standardized data instead of min-max scaling (by default it is min-max), use -scale argument:
python 01_evaluate.py kdd GAN -g 0 -scale standard
```

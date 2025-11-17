# ResNet-Project


## Questions/Goal

What is a ResNet ? How can we construct deeper network ? --> Application to microscopic fungi image classification

## How to launch the project

In the terminal : 

*python3 train_test.py* to produce the raw results

*python3 plots.py* to render the plots

plots et results can be found in the corresponding files 

## Strcuture 

## Project Structure

### Data Processing
- **data_import.py**  
  Handles loading and basic preprocessing of the raw image dataset.

- **data_split.py**  
  Splits the dataset into training, validation, and test sets.  
  Called automatically when running the main script.

### Model Architectures
- **resnet.py**  
  Contains the implementation of the ResNet model used for comparison.

- **cnn.py**  
  Contains the implementation of your custom CNN baseline.

### Training & Evaluation
- **train_test.py**  
  The central script of the project.  
  It:
  - imports and splits the data,  
  - initializes both models,  
  - trains them,  
  - evaluates their performance,  
  - and saves the results to disk.

### Plot Generation
- **plot.py**  
  Builds the comparison figures using the results produced by train_test.py.

### Output Folders
- **results/**  
  Stores raw output data such as logs, metrics, accuracy/loss curves, etc.

- **plots_comparatifs/**  
  Contains all generated graphs comparing the CNN and ResNet performances.




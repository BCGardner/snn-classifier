# SNN Classifier
This repository implements supervised learning in spiking neural networks for data classification.

Libraries are included under the 'snncls' package for data preprocessing, applicable to spike-based processing. Spiking neural network, neuron and training classes are also defined here.

## Requirements
- Python (2.7)
- NumPy
- Matplotlib
- Scikit-learn

Python-mnist is also used when loading the MNIST database of handwritten digits. Unittests are run using Pytest. This repository has been developed on Ubuntu 16.04.

## Setup
The shell script located at 'bin/snncls_vars.sh' exports the paths required to make the snncls package visible in python. This script can be sourced in bash.

## Example usage
After setup, a good starting point is to visualise the process of 'scanline encoding', where handwritten digits are transformed into spike trains via pixel-intensity scanning. This is useful as the preprocessing step for encoding digits as input spikes in an SNN. This process may be visualised under 'projects/scanline-encoder' using the following command:

`$ ./main.py`

Optional command line arguments can be passed to main.py: for a complete list use the '-h' flag.

Examples of multilayer SNN training scripts may be found under 'projects/multilayer-classifier', such as 'main_mnist.py' when training an SNN on the MNIST dataset.

## License
Code released under the GNU General Public License v3.

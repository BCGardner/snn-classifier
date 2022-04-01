# SNN Classifier
This repository implements supervised learning in multilayer spiking neural networks for data classification.

Libraries are included under the 'snncls' package for data preprocessing, applicable to spike-based processing. Spiking neural network, neuron and training classes are also defined here.

## Requirements
- Python3
- NumPy
- Matplotlib
- Python-mnist
- Scikit-learn
- Scikit-image

## Example usage
After setup, a good starting point is to visualise the process of 'scanline encoding', where handwritten digits are transformed into spike trains via pixel-intensity scanning. This is useful as the preprocessing step for encoding digits as input spikes in an SNN. This process may be visualised under 'projects/scanline-encoder' using the following command:

`$ ./main.py`

Optional command line arguments can be passed to main.py: for a complete list use the '-h' flag. MNIST digits being transformed into, for example, eight spike trains via scanline encoding can be visualised using the following command:

`$ ./transform.py -p -s 8`

where the argument `-s` specifies the number of randomly-oriented scanlines used to encode the digit.

Supervised training of feedforward, multilayer SNNs may be found under 'projects/softmax-classifier'. This project demonstrates how datasets containing real-valued features can be transformed into spatiotemporal spike patterns, suitable for processing in SNNs. This project also applies backpropagation to training SNNs to classify datasets based on first-to-spike output responses, within just a few 10's of ms. Examples include classifying the benchmark iris and wisconsin datasets (`stratified_kfold.py`), MNIST when digits are encoded as spike latencies (`mnist_latencies.py`), MNIST when efficiently encoded using scanlines (`mnist_scan.py`) and also experimental work examining radon transformations for dimensionality reduction (`mnist_radon.py`).

## License
Code released under the GNU General Public License v3.

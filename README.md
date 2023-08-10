# Bidomain Modeling Paradigm for Pansharpening
- Code for the paper: "Bidomain Modeling Paradigm for Pansharpening", ACM MM 2023.
- State-of-the-art (SOTA) performance on the [PanCollection](https://github.com/liangjiandeng/PanCollection) of remote sensing pansharpening.

## Method
### pansharpening
![](.\images\head.pdf)
Pansharpening is a challenging low-level vision task whose aim is to fuse LRMS (low-resolution multispectral image) and PAN (panchormatic image) to get HRMS (high-resolution multispectral image).
### BiMPan
#### Overall Structure
![](.\images\overall.pdf)
We empoly a bidomain paradigm for BiMPan, _i.e._, BLSM (Band-Aware Local Specificity Modeling) branch to extract local features and FGDR (Fourier Global Detail Reconstruction) branch to extract global features.
#### BLSM
![](.\images\ADK.pdf)
BLSM branch applies adaptive convolution to explore the local uniqueness of each band.
#### FGDR
![](.\images\Fourier.pdf)
FDGR branch applies convolution in Fourier domain to embracing global information while benefiting the disentanglement of image degradation.
## Experiment results
- Quantitative evalutaion results on WV3 datasets of PanCollection.
![](.\images\results.PNG)
- Visual results on WV3 datasets of PanCollection.
![](.\images\WV3_RR)
![](.\images\WV3_FR)
# Get Strarted
## Dataset
- Datasets for pansharpening: [PanCollection](https://github.com/liangjiandeng/PanCollection). The downloaded data can be placed everywhere because we do not use relative path. Besides, we recommend the h5py format, as if using the mat format, the data loading section needs to be rewritten.
## Denpendcies
- Python 3.10 (Recommend to use Anaconda)
- Pytorch 2.0
- NVIDIA GPU + CUDA
- Python packages: pip install numpy scipy h5py torchsummary
## Code
Training and testing codes are in the current folder.
- The code for training is in main.py, while the code for testing test.py.
- For training, you need to set the file_path in the main function, adopt t your train set, validate set, and test set as well. Our code train the .h5 file, you may change it through changing the code in main function.
- As for testing, you need to set the path in both main and test function to open and load the file.
# Citation
Coming soon.
# Citation
We are glad to hear from you. If you have any questions, please feel free to contact caolucas082@gmail.com.

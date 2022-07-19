## About the Environment Setup
<details open="open">
  <summary></summary>

### Step 1
First of all, clone this repository to your local machine and access the main dir via the following command:
```
git clone https://github.com/awsm-research/VulRepair.git
cd VulRepair
```

### Step 2
If having python3 installed in your environment, you may skip this step.
  
Otherwise, please install python 3.9.7 <a href="https://www.python.org/downloads/release/python-397/">here</a>.

### Step 3
Then, install the python dependencies via the following command:
```
pip install transformers
pip install torch
pip install numpy
pip install tqdm
pip install pandas
pip install tokenizers
pip install datasets
git clone https://github.com/wkentaro/gdown.git
cd gdown
pip install .
cd ..
```

* We highly recommend you check out this <a href="https://pytorch.org/">installation guide</a> for the "torch" library so you can install the appropriate version on your device.
  
* To utilize GPU (optional), you also need to install the CUDA library, you may want to check out this <a href="https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html">installation guide</a>.
  
* <a href="https://www.python.org/downloads/release/python-397/">Python 3.9.7</a> is recommended, which has been fully tested without issues.
 
</details>

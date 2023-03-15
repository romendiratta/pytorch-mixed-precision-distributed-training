# pytorch-mixed-precision-distributed-training

The goal of this repository is to demonstate how to speed up model training in PyTorch. Two methods are tested, mixed precision training and distributed training. For implementation, PyTorch is used. Automatic Mixed Precision (AMP) is used for mixed precision training and Distributed Data Paralell (DDP) is used for distributed training.

<br>

## Setup
### Hardware
For training, AWS g5.2xlarge instances were used with 8 vCPUs, 32 GiB of memory, and a 24 GiB NVIDIA A10G Tensor Core GPU.
### Software
These machines were provisioned with NVIDIA deep learning AMIs and training was executed within the latest pytorch docker image (nvcr.io/nvidia/pytorch:23.02-py3). 
### Data
To train the models the ImageNet training and validation sets were used. These were saved to an AWS EBS volume and attached to each instance.

A PyTorch data loader was used to feed data into the model. Each image was resized to 128 x 128 and normalized against the ImageNet mean and standard deviation. Training used 6 workers to feed data to the model, while validation used only a single process.
### Model 
The model used is resnet18. The fully connected final layer is modified to output 1000 classes.

<br>

## Results
### Baseline Training
In the baseline notebook, a basic PyTorch training and validation loop are setup. Training for one epoch took ~29 minutes.

### Mixed Precision Training (AMP)
In the AMP notebook, the same PyTorch training and validation loops are used from the baseline. They are modified so that the foward pass and loss calculation are contained with an <code>autocast()</code> context manager which will try to modify any Tensors that can be changed to a lower precision. A gradient scaler <code>(GradScaler)</code>is also used to ensure that the gradients are scaled correctly when performing backprogation.
Training for one epoch took ~24 minutes.

### Distributed Training (DDP)
In both DDP notebooks, again the same the same PyTorch training and validation loops are used from the baseline. Instead PyTorch DDP is first intialized in each notebook respectively. Next a <code>DistributedSampler</code> is used to ensure that data is spread across each node during training. Finally, the models are wrapped in <code>DistributedDataParallel</code> to sync the models during training. Training for one epoch took ~18 minutes.

### Conclusion
It is apparent that mixed precision training can provided some speedup as the training time dropped by ~19 percent. In the case of distributed training there was a speedup of ~47%. This is what is expected as the data was now split across two nodes while the models trained in paralell. A perfect ~50 speedup is not expected because of overhead. 





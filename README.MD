# NEW APPROACH TO FINE-TUNE CUSTOM MODELS

I tried to do this using miniconda, but some communication problems with the GPU were encountered. It will be studied in the future.

To start working with it, create a virtual environment and install the requirements.txt. 

python:3.12.2

Distributed training using multiple GPUs is not possible due to GPU version (2070). 
With newer GPUs try to run fsdp.py 
(Thanks to @RonanKMcGovern on GitHub).


```
pip install flash-attn --no-build-isolation
export HF_HUB_ENABLE_HF_TRANSFER=True
```


## Last attempt

The last try is in finetune.py. It will train a DistrilBert model for detecting CVEs vulnerabilities. 

You can test this model using the test.py file.

```
torchrun --nproc_per_node=4 finetune.py
```
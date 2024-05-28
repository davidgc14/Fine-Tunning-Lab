# NEW APPROACH TO FINE TUNE CUSTOM MODELS

Tried to do it using miniconda, but some gpu communication troubles were found. Will be studied in the future.

To start working with it, create a virtual enviroment and install the requirements.txt. 

python:3.12.2

Running only pp version. For other versions, CUDA>11.6 is needed and also need to run next command:

```
pip install flash-attn --no-build-isolation
export HF_HUB_ENABLE_HF_TRANSFER=True
```

Thanks to @RonanKMcGovern on GitHub. 

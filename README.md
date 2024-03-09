# mobile-diffusion

To clone the repository and all submodules:

First apply settings to git to allow greater memory:

```
git config --global pack.windowMemory "100m"
git config --global pack.packSizeLimit "100m"
git config --global pack.threads "1"
```

Then clone repository with all submodules
```
git clone --recurse-submodules https://github.com/davidw0311/mobile-diffusion.git
```

if submodules are missing, or the above commands were not run properly, run
```
git submodule update --init --recursive
```

Initialize a conda environment, then run
```
cd diffusers
pip install -e .
```

Install the following dependencies
```
pip install torch
pip install transformers
pip install accelerate
```

run the inference script
```
python test_diffusion.py
```


Editing the code can be done at

```
diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
```

To start the wandb demo the following additional steps are required
```
pip install wandb
pip install fastdownload
pip install fastprogress

wandb login --relogin
```

## Custom files naming conventions
All modified files will follow the convention of 
```<original_file_name>_mdCustom.py```

All modified classes will follow the convention of 
```class <originalClassname>_mdCustom```

After making creating copies of these files, make sure to update the dependencies in by including the custom files or classes.

[diffusers/src/diffusers/\_\_init\_\_.py](diffusers/src/diffusers/__init__.py)

[diffusers/src/diffusers/pipelines/\_\_init\_\_.py](diffusers/src/diffusers/pipelines/__init__.py)

[diffusers/src/diffusers/pipelines/stable_diffusion/\_\_init\_\_.py](diffusers/src/diffusers/pipelines/stable_diffusion/__init__.py)






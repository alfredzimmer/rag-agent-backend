# This is the python microservice api endpoint
conda create -n pyapi_local python=3.13
conda activate pyapi_local
pip install -r requirements.txt
**if failed because of version conflit, try to update requirements.txt as below**
pip freeze > requirements.txt  

# To update requirements.txt
pip install pip-tools  
put the new library's name in requirements.in  
pip-compile --upgrade (this can take a while)
requirements.txt should be updated  

# To train model using pytorch
Please import pytorch-training-header.py at the beginnning of your module

# Deepeval commands
```
deepeval set-ollama [model_name]
deepeval unset-ollama
```
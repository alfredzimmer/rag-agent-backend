# This is the python microservice api endpoint
conda create -n pyapi_local python=3.13
conda activate pyapi_local
pip install -r requirements.txt
pip freeze > requirements.txt
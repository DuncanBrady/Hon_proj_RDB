import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file) 

data_path = config.get("data_path", "data/")
model_opts = config.get("model", "models/") 

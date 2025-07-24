import yaml

with open('C:\\Users\\rdbra\\Documents\\honoursProject\\code_base\\Hon_proj_RDB\\GRN_work\\src\\train\\config.yaml', "r") as file:
    config = yaml.load(file, Loader=yaml.Loader)

data_path = config.get("data_path", "data/")
models = config.get("models")
encoders = config.get("encoders", {})
decoders = config.get("decoders", {})
encoder_decoder_pairs = config.get("encoder_decoder_pairs", {})


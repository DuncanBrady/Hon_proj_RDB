import yaml

with open('C:\\Users\\rdbra\\Documents\\honoursProject\\code_base\\Hon_proj_RDB\\GRN_work\\src\\train\\config.yaml', "r") as file:
    config = yaml.load(file, Loader=yaml.Loader)

data_path = config.get("data_path", "data/")
models = config.get("models")
encoders = config.get("encoders", {})
decoders = config.get("decoders", {})
encoder_decoder_pairs = config.get("encoder_decoder_pairs", {})
loss_functions = config.get("loss_functions", {})

if __name__ == "__main__":
    print("Configuration loaded successfully.")
    print(f"Data path: {data_path}")
    print(f"Models: {models}")
    print(f"Encoders: {encoders}")
    print(f"Decoders: {decoders}")
    print(f"Encoder-Decoder Pairs: {encoder_decoder_pairs}")
    print(f"Loss Functions: {loss_functions}")
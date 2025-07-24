# This is the training script that will be used to train multiple autoencoders.
import optparse, sys, os
from src.train.config import models, encoders, decoders, data_path
def parse_opts():

    try:
        parser = optparse.OptionParser()
        parser.add_option("-f", "--file", dest="file",default=None, help="Specify the dataset file to use for training (e.g., 'mnist', 'cifar10')")
        parser.add_option("-o", "--output", dest="output", default=None, help="Specify the output directory for saving the trained model, logs, and results")
        parser.add_option("-m", "--model", dest="model", default=None, help="Specify the model to train (e.g., 'autoencoder', 'variational_autoencoder'), cannot be used with --encoder or --decoder")
        parser.add_option("-e", "--encoder", dest="encoder", default=None, help="Specify the encoder architecture (e.g., 'simple', 'resnet'), must be used with --decoder")
        parser.add_option("-d", "--decoder", dest="decoder", default=None, help="Specify the decoder architecture (e.g., 'simple', 'resnet'), must be used with --encoder")
        parser.add_option("-l", "--loss", dest="loss_function", default=None, help="Specify the loss function to use (e.g., 'mse', 'bce')")
        parser.add_option("-t", "--test", action="store_true", dest="test", default=False, help="Use this option to run the script in test mode, which will not perform any training but will check the configuration and print the options")
    except optparse.BadOptionError as err:
        print(str(err))
        sys.exit(2)

    return parser

def check_opts(parser):
    opts, _, = parser.parse_args()
    if not opts.file and not opts.test:
        print("Error: The --file option is required.")
        parser.print_help()
        sys.exit(2)

    if not opts.output and not opts.test:
        print("Error: The --output option is required.")
        parser.print_help()
        sys.exit(2)

    if opts.model and (opts.encoder or opts.decoder) and not opts.test:
        print("Error: The --model option cannot be used with --encoder or --decoder.")
        parser.print_help()
        sys.exit(2)

    if not opts.model and (not opts.encoder or not opts.decoder) and not opts.test:
        print("Error: If using --encoder, you must also specify --decoder, and vice versa.")
        parser.print_help()
        sys.exit(2)

def check_model_config(model_name=None):
    if model_name and model_name not in model_config:
        print(f"Error: Model '{model_name}' is not defined in the configuration.")
        sys.exit(2)
    return models.get(model_name, {})

def train_model(model_name=None, encoder_name=None, decoder_name=None, loss=None):
    #Training logic will go here
    if model_name:
        print(f"Training model: {model_name}")
    elif encoder_name and decoder_name:
        print(f"Using encoder: {encoder_name} and decoder: {decoder_name}")
    print(f"Using loss function: {loss}")
    # After training model will be saved and results will be logged

    print("Training complete.")
    return

def main():    
    parser = parse_opts()
    check_opts(parser)
    opts, args = parser.parse_args()
    # Check if the dataset file exists
    if opts.file:
        dataset_path = os.path.join(data_path, opts.file)
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file '{opts.file}' does not exist in the specified data path '{data_path}'.")
            sys.exit(2)
    train_model(
        model_name=opts.model,
        encoder_name=opts.encoder,
        decoder_name=opts.decoder,
        loss=opts.loss_function
    )
    return
    
if __name__ == "__main__":
    main()
    sys.exit(0)
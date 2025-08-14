# This is the training script that will be used to train multiple autoencoders.
import optparse, sys, os
from src.train import config as train_config
#import config as train_config
def parse_opts():

    try:
        parser = optparse.OptionParser()
        parser.add_option("-f", "--file", dest="file",default=None, 
                        help="Specify the dataset file to use for training (e.g., 'mnist', 'cifar10')")
        parser.add_option("-o", "--output", dest="output", default=None, 
                        help="Specify the output directory for saving the trained model, logs, and results")
        parser.add_option("-m", "--model", dest="model", default=None, 
                        help="Specify the model to train (e.g., 'autoencoder', 'variational_autoencoder'), cannot be used with --encoder or --decoder")
        parser.add_option("-e", "--encoder", dest="encoder", default=None, 
                        help="Specify the encoder architecture (e.g., 'simple', 'resnet'), must be used with --decoder")
        parser.add_option("-d", "--decoder", dest="decoder", default=None, 
                        help="Specify the decoder architecture (e.g., 'simple', 'resnet'), must be used with --encoder")
        parser.add_option("-l", "--loss", dest="loss_function", default="weighted_huber", 
                        help="Specify the loss function to use (e.g., 'mse', 'bce')")
        parser.add_option("--lr", "--learning_rate", dest="learning_rate", default=0.001, type=float,
                        help="Specify the learning rate for training (default: 0.001)")
        parser.add_option("--ep", "--epochs", dest="epochs", default=10, type=int,
                        help="Specify the number of epochs for training (default: 10)")
        parser.add_option("-t", "--test", action="store_true", dest="test", default=False, 
                        help="Use this option to run the script in test mode, which will not perform any training but will check the configuration and print the options")
    except optparse.BadOptionError as err:
        print(str(err))
        sys.exit(2)

    return parser

def check_req_opts(parser):

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

    if opts.epochs <= 0:
        print("Error: The number of epochs must be a positive integer.")
        parser.print_help()
        sys.exit(2)

    if opts.learning_rate <= 0:
        print("Error: The learning rate must be a positive number.")
        parser.print_help()
        sys.exit(2)

def check_dataset_file(file_path):
    dataset_path = os.path.join(train_config.data_path, file_path)
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{file_path}' does not exist in the specified data path '{dataset_path}'.")
        sys.exit(2)

def check_output_dir(output_dir):
    output_dir = os.path.join(train_config.data_path, output_dir)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error: Could not create output directory '{output_dir}'. {e}")
            sys.exit(2)
    else:
        '''
        print(f"Output directory '{output_dir}' already exists. Do you want to save results there ? (y/n)")
        response = input().strip()
        if response != 'y':
            new_path = input("Please specify a different output directory: ")
            check_output_dir(new_path)
        else: 
            print(f'Results will be saved in {output_dir}, some files may be overwritten.')
            '''
        print(f"Output directory '{output_dir}' is ready for saving results.")


def check_model_config(model_name=None):
    if model_name and model_name not in train_config.models:
        print(f"Error: Model '{model_name}' is not defined in the configuration.")
        sys.exit(2)
    return train_config.models.get(model_name, {})

def check_encoder_decoder(encoder_name=None, decoder_name=None):
    if encoder_name and encoder_name not in train_config.encoders:
        print(f"Error: Encoder '{encoder_name}' is not defined in the configuration.")
        sys.exit(2)
    if decoder_name and decoder_name not in train_config.decoders:
        print(f"Error: Decoder '{decoder_name}' is not defined in the configuration.")
        sys.exit(2)
    if (encoder_name,decoder_name) not in train_config.encoder_decoder_pairs:
        print(f"Error: The combination of encoder '{encoder_name}' and decoder '{decoder_name}' is not valid.")
        sys.exit(2)

def check_loss_function(loss_function=None):
    if loss_function not in train_config.loss_functions:
        print(f"Error: Loss function '{loss_function}' is not defined in the configuration.")
        sys.exit(2)
    return loss_function

def train_model(model_name=None, encoder_name=None, decoder_name=None, loss=None, 
                learning_rate=None, epochs=None):
    #Training logic will go here
    if model_name:
        print(f"Training model: {model_name}")
    elif encoder_name and decoder_name:
        print(f"Using encoder: {encoder_name} and decoder: {decoder_name}")
    print(f"Using loss function: {loss}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of Epochs: {epochs}")
    # After training model will be saved and results will be logged

    print("Training complete.")
    return

def main():    
    parser = parse_opts()
    check_req_opts(parser)
    opts, args = parser.parse_args()
    check_dataset_file(opts.file)
    check_output_dir(opts.output)
    # Check if the output directory exists, if not create i
    if opts.model:
        check_model_config(opts.model)
    elif opts.encoder and opts.decoder:
        check_encoder_decoder(opts.encoder, opts.decoder)
    check_loss_function(opts.loss_function)
    train_model(
        model_name=opts.model,
        encoder_name=opts.encoder,
        decoder_name=opts.decoder,
        loss=opts.loss_function,
        learning_rate=opts.learning_rate,
        epochs=opts.epochs
    )
    sys.exit(0)
    
if __name__ == "__main__":
    main()
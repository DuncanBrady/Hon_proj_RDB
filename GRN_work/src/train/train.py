# This is the training script that will be used to train multiple autoencoders.
import getopt, sys

def print_help():
    print("Usage: python train.py -f <input_directory> -o <output_directory> (-m <model> | -e <encoder> -d <decoder>) [-l <loss_function>]")
    print("Required Args:")
    print("  -f, --file      Specify the dataset file to use for training (e.g., 'mnist', 'cifar10')")
    print("  -o, --output    Specify the output directory for saving the trained model, logs, and results")
    print("\tModel Specification:(choose one of the following)")
    print("\t -m, --model     Specify the model to train (e.g., 'autoencoder', 'variational_autoencoder'), cannot be used with --encoder or --decoder")
    print("\t\tOR:")
    print("\t -e, --encoder   Specify the encoder architecture (e.g., 'simple', 'resnet'), must be used with --decoder")
    print("\t -d, --decoder   Specify the decoder architecture (e.g., 'simple', 'resnet'), must be used with --encoder")
    print("Optional Args:")
    print("  -l, --loss      Specify the loss function to use (e.g., 'mse', 'bce')")

def parse_opts():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:o:m:e:d:l:", ["help", "file=", "output=", "model=", "encoder=", "decoder=", "loss="])
        if ("-h","") in opts or ("--help","") in opts:
            print_help()
            sys.exit(0)
    except getopt.GetoptError as err:
        print(str(err))
        print_help()
        sys.exit(2)

    return opts

def check_opts(opts):
    # Check for mutual exclusivity
    if opts.get("-m") and (opts.get("-e") or opts.get("-d")):
        print("Error: Cannot specify both model and encoder/decoder.")
        print_help()
        sys.exit(2)
    if (opts.get("-e") and not opts.get("-d")) or (opts.get("-d") and not opts.get("-e")):
        print("Error: Both encoder and decoder must be specified.")
        print_help()
        sys.exit(2)

def main():
    opts = parse_opts()
    print(type(opts))
    print(opts)
    check_opts(opts)

    # Extract options
    file = opts.get("-f") or opts.get("--file")
    output = opts.get("-o") or opts.get("--output")
    model = opts.get("-m") or None
    encoder = opts.get("-e") or None
    decoder = opts.get("-d") or None
    loss_function = opts.get("-l") or "mse"  # Default to 'mse' if not specified

    print(f"Training with dataset: {file}")
    print(f"Output directory: {output}")
    
    if model:
        print(f"Choosing model to train...{model}")
    else:
        print(f"Choosing encoder-decoder architecture...Encoder: {encoder}, Decoder: {decoder}")

    print(f"Using loss function: {loss_function}")

    
if __name__ == "__main__":
    main()

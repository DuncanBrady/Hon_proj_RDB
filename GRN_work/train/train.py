# This is the training script that will be used to train multiple autoencoders.
import getopt, sys

usage = "Usage: python train.py -f <input_directory> -o <output_directory> (-m <model> | -e <encoder> -d <decoder>) [-l <loss_function>]"

try:
    arglist = sys.argv[1:]
    opts, args = getopt.getopt(arglist, "hm:f:o:l:e:d:", ["help", "model=", "file=", "output=", "loss=", "encoder=", "decoder="])

    if len(opts) == 0 or ("-h", "") in opts or ("--help", "") in opts:
        print(usage)
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
        sys.exit(0)
    if ("-m", "") in opts or ("--model", "") in opts:
        if ("-e", "") in opts or ("--encoder", "") in opts or ("-d", "") in opts or ("--decoder", "") in opts:
            print("Error: Cannot use --model with --encoder or --decoder. Please choose one.")
            sys.exit(1)
        print("Choosing model to train...")
        exit(0)
    elif ("-e", "") in opts or ("--encoder", "") in opts and ("-d", "") in opts or ("--decoder", "") in opts:
        print("Choosing encoder-decoder architecture...")
        exit(0)
    else:
        print("Error: Incorrect arguments provided. Please use -h or --help for usage instructions.")
        print(usage)
        print("Exiting...")
        sys.exit(1)
except getopt.GetoptError as err:
    print(f"Error: {err}")
    print(usage)
    sys.exit(2)
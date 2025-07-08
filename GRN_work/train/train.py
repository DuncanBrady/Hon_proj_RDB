# This is the training script that will be used to train multiple autoencoders.
import getopt, sys


arglist = sys.argv[1:]
opts, args = getopt.getopt(arglist, "h", ["help"])
if len(opts) == 0 or ("-h", "") in opts or ("--help", "") in opts:
    print("Usage: python train.py [options]")
    print("Options:")
    print("  -h, --help      Show this help message and exit")
    print("-m, --model     Specify the model to train (e.g., 'autoencoder', 'variational_autoencoder')")
    print("-d, --dataset   Specify the dataset to use for training (e.g., 'mnist', 'cifar10')")
    print("-l, --loss      Specify the loss function to use (e.g., 'mse', 'bce')")
    print("-e, --encoder   Specify the encoder architecture (e.g., 'simple', 'resnet')")
    print("-d, --decoder   Specify the decoder architecture (e.g., 'simple', 'resnet')")
    sys.exit(0)
import optparse
import config as train_config
if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("--option1", dest="option1", default="default_value1", help="Description for option1")
    parser.add_option("--option2", dest="option2", default="default_value2", help="Description for option2")
    parser.add_option("--c","--check", dest="c", action="store_true", default=False, help="Description for option3")
    (options, args) = parser.parse_args()
    print(type(options), type(args))
    print("Option 1:", options.option1)
    print("Option 2:", options.option2)
    print("Option 3:", options.c)
    print("Arguments:", args)
    print("Model Config:", train_config.model_opts)/
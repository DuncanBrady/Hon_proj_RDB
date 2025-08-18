import pytest
@pytest.fixture
def validFilePaths():
    return ["train.py", 
            "--file", "test_data.csv",
            "--output", "test_output"]

@pytest.fixture
def validModel(validFilePaths):
    return validFilePaths + [
            "--model", "convautoencoder"
            ]

@pytest.fixture
def validEncoderDecoder(validFilePaths):
    return validFilePaths + [
            "--encoder", "sequential", 
            "--decoder", "sequential",
            ]

@pytest.fixture
def validLearningRate(validModel):
    return validModel + [
            "--learning_rate", "0.005"
            ]
@pytest.fixture
def validLossFunction(validModel): 
    return validModel + [
            "--loss", "weighted_huber"
            ]

@pytest.fixture
def validEpochs(validModel):
    return validModel + [
            "--epochs", "50"
            ]

@pytest.fixture
def invalidModel(validFilePaths):
    return validFilePaths + [
            "--model", "invalid_model",
            ]
@pytest.fixture
def invalidEncoder(validFilePaths):
    return validFilePaths + [
            "--encoder", "invalid_encoder",
            "--decoder", "sequential",
            ]

@pytest.fixture
def invalidDecoder(validFilePaths):
    return validFilePaths + [
            "--decoder", "invalid_decoder",
            "--encoder", "sequential",
            ]

@pytest.fixture
def invalidEncoderDecoder(validFilePaths):
    return validFilePaths + [
            "--encoder", "invalid_encoder",
            "--decoder", "invalid_decoder",
            ]

@pytest.fixture
def invalidEncoderValidDecoder(validFilePaths):
    return validFilePaths + [
            "--decoder", "sequential",
            "--encoder", "invalid_encoder",
            ]
@pytest.fixture
def invalidDecoderValidEncoder(validFilePaths):
    return validFilePaths + [
            "--encoder", "invalid_encoder",
            "--decoder", "sequential",
            ]
@pytest.fixture
def nonMatchEncoderDecoderPair(validFilePaths):
    return validFilePaths + [
            "--encoder", "sequential",
            "--decoder", "valid_decoder",
            ]

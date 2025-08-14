import sys
import pytest
from src.train.train import main

class TestTrain:
    def test_help_option_long(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['train.py', '--help', '--test'])
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Usage: train.py" in captured.out
        assert "Options:" in captured.out

    def test_help_option_short(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['train.py', '-h', '--test '])
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Usage: train.py" in captured.out
        assert "Options:" in captured.out

    def test_model_option(self, capsys, monkeypatch, validModel):
        monkeypatch.setattr(sys, 'argv', validModel)
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Training model: convautoencoder" in captured.out

    def test_encoder_decoder_option(self, capsys, monkeypatch, validEncoderDecoder):
        monkeypatch.setattr(sys, 'argv', validEncoderDecoder)
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Using encoder: sequential and decoder: sequential" in captured.out

    def test_loss_function_option(self, capsys, monkeypatch, validLossFunction):
        monkeypatch.setattr(sys, 'argv', validLossFunction)
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Using loss function: weighted_huber" in captured.out   

    def test_learning_rate_option(self, capsys, monkeypatch, validLearningRate):
        monkeypatch.setattr(sys, 'argv', validLearningRate)
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert f"Learning rate: 0.005" in captured.out

    def test_epochs_option(self, capsys, monkeypatch, validEpochs):
        monkeypatch.setattr(sys, 'argv', validEpochs)
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Number of Epochs: 50" in captured.out

    def test_invalid_model_option(self, capsys, monkeypatch, invalidModel):
        monkeypatch.setattr(sys, 'argv', invalidModel)
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Error: Model 'invalid_model' is not defined in the configuration." in captured.out

    def test_invalid_encoder_option(self, capsys, monkeypatch, invalidEncoder):
        monkeypatch.setattr(sys, 'argv', invalidEncoder)
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Error: Encoder 'invalid_encoder' is not defined in the configuration." in captured.out

    def test_invalid_decoder_option(self, capsys, monkeypatch, invalidDecoder):
        monkeypatch.setattr(sys, 'argv', invalidDecoder)
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Error: Decoder 'invalid_decoder' is not defined in the configuration." in captured.out  
    
    def test_imcomp_encoder_decoder_pair(self, capsys, monkeypatch, nonMatchEncoderDecoderPair):
        monkeypatch.setattr(sys, 'argv', nonMatchEncoderDecoderPair)
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Error: The combination of encoder 'sequential' and decoder 'valid_decoder' is not valid." in captured.out
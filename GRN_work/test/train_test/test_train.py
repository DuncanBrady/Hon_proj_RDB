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

    def test_model_option(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['train.py', '-m', 'test1', '--test'])
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Training model: test1" in captured.out

    def test_encoder_decoder_option(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['train.py', '-e', 'sequential', '-d', 'sequential', '--test'])
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Using encoder: sequential and decoder: sequential" in captured.out

    def test_loss_function_option(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['train.py', '-l', 'weighted_huber', '--test'])
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Using loss function: weighted_huber" in captured.out   

    def test_learning_rate_option(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['train.py', '--lr', '0.01', '--test'])
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert f"Learning rate: 0.01" in captured.out

    def test_epochs_option(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['train.py', '--epochs', '10', '--test'])
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Number of Epochs: 10" in captured.out
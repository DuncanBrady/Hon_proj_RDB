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
        monkeypatch.setattr(sys, 'argv', ['train.py', '-e', 'simple', '-d', 'resnet', '--test'])
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Using encoder: simple and decoder: resnet" in captured.out
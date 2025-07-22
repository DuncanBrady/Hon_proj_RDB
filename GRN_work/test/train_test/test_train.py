import sys
import pytest
from train.train import main


class TestTrain:
    def test_help_option_long(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['train.py', '--help'])
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Usage: python train.py" in captured.out
        assert "Optional Args:" in captured.out

    def test_help_option_short(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['train.py', '-h'])
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Usage: python train.py " in captured.out
        assert "Optional Args:" in captured.out
'''
    def test_model_option(self, capsys):
        with pytest.raises(SystemExit):
            main(['-m', 'test1'])
        captured = capsys.readouterr()
        assert "Choosing model to train...test1" in captured.out

    def test_encoder_decoder_option(self, capsys):
        with pytest.raises(SystemExit):
            main(['-e', 'simple', '-d', 'resnet'])
        captured = capsys.readouterr()
        assert "Choosing encoder-decoder architecture..." in captured.out
        '''
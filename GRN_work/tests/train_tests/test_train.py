import pytest
from Hon_proj_RDB.GRN_work.train.train import main

class TestTrain:
    def test_help_option(self, capsys):
        with pytest.raises(SystemExit):
            main(['-h'])
        captured = capsys.readouterr()
        assert "Usage: python train.py" in captured.out
        assert "Required Args:" in captured.out
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
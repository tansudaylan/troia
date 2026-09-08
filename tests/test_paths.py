import os

from troia.main import retr_pathtroy


def test_retr_pathtroy_creates_expected_directories(monkeypatch, tmp_path):
    monkeypatch.setenv('TROIA_DATA_PATH', str(tmp_path / 'troia'))

    dictpath = retr_pathtroy(strgextn='SyntheticPopulation_TESS')

    assert os.path.isdir(dictpath['pathbase'])
    assert os.path.isdir(dictpath['pathdatapipe'])
    assert os.path.isdir(dictpath['pathvisupipe'])
    assert os.path.isdir(dictpath['pathpopl'])
    assert os.path.isdir(dictpath['pathvisucnfg'])
    assert os.path.isdir(dictpath['pathdatacnfg'])
    assert os.path.normpath(dictpath['pathpopl']).endswith(
        os.path.normpath('SyntheticPopulation_TESS')
    )
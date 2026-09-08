import importlib


def test_legacy_kartik_eli_is_archived():
    try:
        importlib.import_module('troia.kartik_eli')
    except ImportError:
        return
    raise AssertionError('troia.kartik_eli should be excluded from the supported API and fail on import.')

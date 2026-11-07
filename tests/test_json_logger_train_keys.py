from unittest import mock

import pytest

from stoix.utils.logger import JsonLogger, LogEvent


@pytest.mark.parametrize("key", JsonLogger._METRICS_TO_LOG)
def test_train_eval_key_is_written(key):
    if key.startswith("train/"):
        return
    logger = JsonLogger.__new__(JsonLogger)  # skip file setup
    logger.logger = mock.Mock()
    logger.log_stat(f"train/{key}", 1.0, 10, 2, LogEvent.EVAL)
    written = logger.logger.write.call_args.args
    assert written[1] == "_".join(reversed(f"train/{key}".split("/")))

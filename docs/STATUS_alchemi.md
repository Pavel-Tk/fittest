# ALCHEMI integration status

- [00:05] `trustswarm/models/mace_batched.py` (BatchedMACE via Toolkit MACEWrapper) + `models/toolkit_io.py` written; VM check pending (Colab kernel exec queue is slow).
- [00:05] `trustswarm/hook.py` (DisagreementMonitor + DisagreementHook) written; local CPU tests `tests/test_hook.py`: 5 passed, 2 skipped (torch/nvalchemi absent locally).

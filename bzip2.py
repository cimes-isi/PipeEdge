"""Hook functions for bzip2 compression/decompression."""
import logging
import os
import pickle
import shutil
import subprocess
import numpy as np
import torch

# User-specified bzip2, e.g., to switch implementations for binary diversification
ENV_BZIP2_BINARY: str = 'BZIP2_BINARY'

logger = logging.getLogger(__name__)

def _get_bzip2_binary() -> str:
    bzip2 = os.getenv('ENV_BZIP2_BINARY')
    if bzip2 is None:
        bzip2 = shutil.which('bzip2')
    if bzip2 is None:
        raise RuntimeError('Failed to find bzip2')
    return bzip2

def forward_pre_bzip2_decompress(_module, inputs):
    """Decompresss `inputs` to their original type using a bzip2 subprocess and pickle."""
    assert isinstance(inputs, tuple)
    assert len(inputs) == 1
    assert isinstance(inputs[0], torch.Tensor)
    bzip2 = _get_bzip2_binary()
    bytes_in = inputs[0].numpy().tobytes()
    args = [bzip2, '-cd']
    logger.info("%s: decompress", bzip2)
    bzip_dec = subprocess.run(args, capture_output=True, input=bytes_in, check=True)
    return (pickle.loads(bzip_dec.stdout),)

def forward_hook_bzip2_compress(_module, _inputs, outputs):
    """Compresss `outputs` into a uint8 tensor using pickle and a bzip2 subprocess."""
    bzip2 = _get_bzip2_binary()
    bytes_out = pickle.dumps(outputs)
    args = [bzip2, '-c']
    logger.info("%s: compress", bzip2)
    bzip_com = subprocess.run(args, capture_output=True, input=bytes_out, check=True)
    return torch.tensor(np.frombuffer(bzip_com.stdout, dtype=np.uint8))

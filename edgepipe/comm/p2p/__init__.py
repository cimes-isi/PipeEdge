"""Peer-to-peer communication module."""
import collections
import queue
import threading
import time
import torch
import torch.distributed as dist
from .util import DistRequestWaitDaemon

# Base tag values
TAG_BASE_DATA = 0
TAG_BASE_CMD = 10

# Offsets which are added to base values above
TAG_TENSOR_COUNT = 0
TAG_TENSOR_DTYPE = 1
TAG_TENSOR_SHAPE_LEN = 2
TAG_TENSOR_SHAPE = 3
TAG_TENSOR = 4

# Ordered set of torch types: https://pytorch.org/docs/stable/tensor_attributes.html
TORCH_TYPES = [ torch.float32,
                torch.float64,
                torch.complex64,
                torch.complex128,
                torch.float16,
                torch.bfloat16,
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.bool ]
TORCH_TYPES_ENUM = collections.OrderedDict()
for i, t in enumerate(TORCH_TYPES):
    TORCH_TYPES_ENUM[t] = i


class ConditionQueue(queue.Queue):
    """A Queue with a public `condition: threading.Condition` variable for synchronization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.condition = threading.Condition()


def _send_tensor(tensor, dst, tag_base, fn_send=dist.send):
    # NOTE: could optimize by packing dtype and shape length into one message
    tensor_dtype = torch.tensor(TORCH_TYPES_ENUM[tensor.dtype], dtype=torch.int)
    tensor_shape_len = torch.tensor(len(tensor.shape), dtype=torch.int)
    tensor_shape = torch.tensor(tensor.shape, dtype=torch.int)
    results = []
    results.append(fn_send(tensor=tensor_dtype, dst=dst, tag=tag_base+TAG_TENSOR_DTYPE))
    results.append(fn_send(tensor=tensor_shape_len, dst=dst, tag=tag_base+TAG_TENSOR_SHAPE_LEN))
    results.append(fn_send(tensor=tensor_shape, dst=dst, tag=tag_base+TAG_TENSOR_SHAPE))
    results.append(fn_send(tensor=tensor, dst=dst, tag=tag_base+TAG_TENSOR))
    return results


def _recv_tensor(src, tag_base):
    tensor_dtype = torch.zeros(1, dtype=torch.int)
    dist.recv(tensor=tensor_dtype, src=src, tag=tag_base+TAG_TENSOR_DTYPE)
    _tensor_dtype = TORCH_TYPES[int(tensor_dtype)]
    tensor_shape_len = torch.zeros(1, dtype=torch.int)
    dist.recv(tensor=tensor_shape_len, src=src, tag=tag_base+TAG_TENSOR_SHAPE_LEN)
    _tensor_shape_len = int(tensor_shape_len)
    tensor_shape = torch.zeros(_tensor_shape_len, dtype=torch.int)
    dist.recv(tensor=tensor_shape, src=src, tag=tag_base+TAG_TENSOR_SHAPE)
    _tensor_shape = [int(x) for x in tensor_shape] # list(map(lambda x: int(x), tensor_shape))
    tensor = torch.zeros(_tensor_shape, dtype=_tensor_dtype)
    dist.recv(tensor=tensor, src=src, tag=tag_base+TAG_TENSOR)
    return tensor


class TensorSendThread(threading.Thread):
    """Thread for sending tensors."""

    def __init__(self, queue_out, dst_rank):
        super().__init__()
        self._queue_out = queue_out
        self._dst_rank = dst_rank
        self._evt_stop_thread = threading.Event()

    def stop(self):
        """Direct the thread to stop."""
        with self._queue_out.condition:
            self._evt_stop_thread.set()
            self._queue_out.condition.notify_all()

    def run(self):
        """Dequeue tensors and send them."""
        while not self._evt_stop_thread.is_set():
            with self._queue_out.condition:
                while self._queue_out.empty():
                    if self._evt_stop_thread.is_set():
                        return
                    self._queue_out.condition.wait()
                tensors = self._queue_out.get(block=False)
                self._queue_out.condition.notify_all()
            # tensors come in pairs, except for the last stage results
            if isinstance(tensors, torch.Tensor):
                tensors = (tensors,)
            assert isinstance(tensors, tuple)
            assert len(tensors) > 0
            assert isinstance(tensors[0], torch.Tensor)
            _tensor_count = len(tensors)
            tensor_count = torch.tensor(_tensor_count, dtype=torch.int)
            dist.send(tensor=tensor_count, dst=self._dst_rank, tag=TAG_BASE_DATA+TAG_TENSOR_COUNT)
            # NOTE: could optimize by only sending dtype once (it's the same for all tensors)
            for tensor in tensors:
                _send_tensor(tensor, self._dst_rank, TAG_BASE_DATA)


class TensorRecvThread(threading.Thread):
    """Thread for receiving tensors."""

    def __init__(self, queue_in, src_rank):
        super().__init__()
        self._queue_in = queue_in
        self._src_rank = src_rank
        self._evt_stop_thread = threading.Event()

    def stop(self):
        """Direct the thread to stop."""
        self._evt_stop_thread.set()

    def run(self):
        """Receive tensors and enqueue them."""
        while True:
            tensor_count = torch.zeros(1, dtype=torch.int)
            ircv_req = dist.irecv(tensor=tensor_count, src=self._src_rank, tag=TAG_BASE_DATA+TAG_TENSOR_COUNT)
            ircv_req_t = DistRequestWaitDaemon(ircv_req)
            ircv_req_t.start()
            while ircv_req_t.is_alive():
                if self._evt_stop_thread.is_set():
                    return
                # TODO: we're basically spinning...
                time.sleep(0.1)
            _tensor_count = int(tensor_count)
            assert _tensor_count > 0
            tensors = ()
            for _ in range(_tensor_count):
                tensor = _recv_tensor(self._src_rank, TAG_BASE_DATA)
                tensors += (tensor,)
            if _tensor_count == 1:
                # At this point, we don't know whether the original data type was a Tensor or Tuple[Tensor] w/ len=1.
                # We'd have to include that information in a separate message to know for sure.
                # For now, it works to reduce to the base case - just a single Tensor.
                tensors = tensors[0]
            # Blocks if queue is full, which then blocks receiving more tensors (as intended)
            # Worker thread must be running to avoid indefinite blocking
            with self._queue_in.condition:
                while self._queue_in.full():
                    self._queue_in.condition.wait()
                self._queue_in.put(tensors)
                self._queue_in.condition.notify_all()


class TensorWorkThread(threading.Thread):
    """Thread for processing tensors."""

    def __init__(self, queue_in, queue_out, callback):
        super().__init__()
        self._queue_in = queue_in
        self._queue_out = queue_out
        self._callback = callback
        self._evt_stop_thread = threading.Event()

    def stop(self):
        """Direct the thread to stop."""
        with self._queue_in.condition:
            self._evt_stop_thread.set()
            self._queue_in.condition.notify_all()

    def run(self):
        """Dequeue, process, enqueue."""
        # Empty inbound queue before stopping
        while True:
            with self._queue_in.condition:
                while self._queue_in.empty():
                    if self._evt_stop_thread.is_set():
                        return
                    self._queue_in.condition.wait()
                tensor_in = self._queue_in.get(block=False)
                self._queue_in.condition.notify_all()
            tensor_out = self._callback(tensor_in)
            if tensor_out is not None:
                # Sender thread must be running to avoid indefinite blocking
                with self._queue_out.condition:
                    while self._queue_out.full():
                        self._queue_out.condition.wait()
                    self._queue_out.put(tensor_out)
                    self._queue_out.condition.notify_all()


class CommandThread(threading.Thread):
    """Thread for receiving commands."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self._evt_stop_thread = threading.Event()

    def stop(self):
        """Direct the thread to stop."""
        self._evt_stop_thread.set()

    def run(self):
        """Listen for commands."""
        while True:
            # contains (1) CMD enumeration and (2) an optional tensor count
            tensor_cmd = torch.zeros(2, dtype=torch.int)
            ircv_req = dist.irecv(tensor=tensor_cmd, tag=TAG_BASE_CMD)
            ircv_req_t = DistRequestWaitDaemon(ircv_req)
            ircv_req_t.start()
            while ircv_req_t.is_alive():
                if self._evt_stop_thread.is_set():
                    return
                # TODO: we're basically spinning...
                time.sleep(0.1)
            cmd = int(tensor_cmd[0])
            _tensor_count = int(tensor_cmd[1])
            tensors = ()
            for _ in range(_tensor_count):
                # it would be nice if we could restrict src to the prior request's src, but the
                # ircv_req "distributed request object" API doesn't document a src property to use
                tensor = _recv_tensor(None, TAG_BASE_CMD)
                tensors += (tensor,)
            self._callback(cmd, tensors)


def init(rank, world_size):
    """Initialize p2p."""
    dist.init_process_group(dist.Backend.GLOO, rank=rank, world_size=world_size)

def shutdown():
    """Shutdown p2p."""
    dist.destroy_process_group()

def cmd_broadcast(cmd, tensors=None):
    """Broadcast a command."""
    if tensors is None:
        tensors = ()
    elif isinstance(tensors, torch.Tensor):
        tensors = (tensors,)
    tensor_cmd = torch.tensor([cmd, len(tensors)], dtype=torch.int)
    reqs = []
    for dst in range(dist.get_world_size()):
        if dst != dist.get_rank():
            reqs.append(dist.isend(tensor_cmd, dst=dst, tag=TAG_BASE_CMD))
            for tensor in tensors:
                reqs += _send_tensor(tensor, dst, TAG_BASE_CMD, fn_send=dist.isend)
    for req in reqs:
        req.wait()
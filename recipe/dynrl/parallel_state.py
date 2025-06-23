from contextlib import contextmanager

import vllm.distributed.parallel_state as ps

_TP_DEVICE_CANDIDATE_GROUP_DICT = dict()


@contextmanager
def initialize_candidate_ctx(tp_size):
    """
    Initialize the candidate context.
    """
    global _TP_DEVICE_CANDIDATE_GROUP_DICT
    yield
    _TP, _PP, _DP = ps._TP, ps._PP, ps._DP
    # Check if _EP exists before including it
    if hasattr(ps, "_EP"):
        _EP = ps._EP
        _TP_DEVICE_CANDIDATE_GROUP_DICT[tp_size] = (_TP, _PP, _DP, _EP)
        ps._EP = None
    else:
        _TP_DEVICE_CANDIDATE_GROUP_DICT[tp_size] = (_TP, _PP, _DP)
    ps._TP = None
    ps._PP = None
    ps._DP = None


@contextmanager
def enter_candidate_ctx(tp_size):
    """
    Initialize the candidate context.
    """
    _CUR_TP, _CUR_PP, _CUR_DP = ps._TP, ps._PP, ps._DP
    # Check if _EP exists before including it
    if hasattr(ps, "_EP"):
        _CUR_EP = ps._EP

    global _TP_DEVICE_CANDIDATE_GROUP_DICT
    ctx_values = _TP_DEVICE_CANDIDATE_GROUP_DICT[tp_size]

    # Unpack based on whether we stored _EP or not
    if len(ctx_values) == 4:
        _TP, _PP, _DP, _EP = ctx_values
        ps._EP = _EP
    else:
        _TP, _PP, _DP = ctx_values

    ps._TP = _TP
    ps._PP = _PP
    ps._DP = _DP
    yield

    # Restore original values
    ps._TP = _CUR_TP
    ps._PP = _CUR_PP
    ps._DP = _CUR_DP
    if hasattr(ps, "_EP"):
        ps._EP = _CUR_EP


def set_default_ctx(tp_size):
    """
    Set the default context.
    """
    global _TP_DEVICE_CANDIDATE_GROUP_DICT
    ctx_values = _TP_DEVICE_CANDIDATE_GROUP_DICT[tp_size]

    # Unpack based on whether we stored _EP or not
    if len(ctx_values) == 4:
        _TP, _PP, _DP, _EP = ctx_values
        if hasattr(ps, "_EP"):
            ps._EP = _EP
    else:
        _TP, _PP, _DP = ctx_values

    ps._TP = _TP
    ps._PP = _PP
    ps._DP = _DP

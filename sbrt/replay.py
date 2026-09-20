"""Synchronized, evaluator-owned streaming replay and profiling."""
import math
import time
import numpy as np
from .detectors import infer


class ProtocolError(RuntimeError):
    pass


class GuardedOnline:
    __slots__ = ("_values", "consumed", "pending", "ended")

    def __init__(self, values):
        self._values = iter(values)
        self.consumed = 0
        self.pending = False
        self.ended = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.pending:
            raise ProtocolError("Consumed another point before yielding a prediction")
        try:
            value = next(self._values)
        except StopIteration:
            self.ended = True
            raise
        self.pending = True
        self.consumed += 1
        return float(value)

    def __len__(self):
        raise ProtocolError("Online horizon is unavailable")

    def __array__(self, *args, **kwargs):
        raise ProtocolError("Online materialization is forbidden")


def replay(records, detector="official_ewma", infer_fn=None, state_observer=None):
    """Yield (record, predictions, timing) without buffering all series.

    The guard tests accidental API misuse, not hostile code introspection.
    Per-tick clock calls add overhead; timings explicitly include it.
    """
    records = iter(records)
    active = {"stream": None, "record": None, "init_seconds": 0.0, "exhausted": False}

    def datasets():
        for record in records:
            previous = active["stream"]
            if previous is not None and (previous.pending or not previous.ended):
                raise ProtocolError("Advanced datasets before finishing the previous stream")
            stream = GuardedOnline(record.online)
            active["stream"], active["record"] = stream, record
            yield record.historical, stream
        active["exhausted"] = True

    # Wrap construction to separate historical initialization from updates.
    def measured_infer(ds):
        from .detectors import make_detector
        yield
        for history, online in ds:
            start = time.perf_counter()
            state = make_detector(detector, history)
            active["init_seconds"] = time.perf_counter() - start
            if state_observer is not None:
                state_observer("initialize", state)
            for x in online:
                score = state.update(x)
                if state_observer is not None:
                    state_observer("update", state)
                yield score
            if state_observer is not None:
                state_observer("complete", state)

    generator = (infer_fn or measured_infer)(datasets())
    if next(generator) is not None or active["stream"] is not None:
        raise ProtocolError("Missing initial handshake or early dataset consumption")
    current = None
    predictions = []
    step_seconds = 0.0
    init_seconds = 0.0
    while True:
        start = time.perf_counter()
        try:
            score = next(generator)
        except StopIteration:
            break
        elapsed = time.perf_counter() - start
        record, stream = active["record"], active["stream"]
        if record is None or not stream.pending:
            raise ProtocolError("Prediction without a new point")
        if current is not record:
            if current is not None:
                if len(predictions) != len(current.online):
                    raise ProtocolError("Incomplete trajectory")
                yield current, np.asarray(predictions), dict(
                    initialization_seconds=init_seconds, update_seconds=step_seconds)
            current, predictions, step_seconds = record, [], 0.0
            init_seconds = active["init_seconds"]
            # First next() also loads the evaluator's next record. Exclude
            # that call from update throughput rather than misattribute I/O.
            elapsed = 0.0
        if stream.consumed != len(predictions) + 1:
            raise ProtocolError("Wrong consumption count")
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ProtocolError("Invalid prediction")
        predictions.append(score)
        stream.pending = False
        step_seconds += elapsed
    if current is not None:
        if len(predictions) != len(current.online) or active["stream"].pending:
            raise ProtocolError("Generator ended before complete predictions")
        yield current, np.asarray(predictions), dict(
            initialization_seconds=init_seconds, update_seconds=step_seconds)
    if not active["exhausted"]:
        raise ProtocolError("Generator did not finish the dataset stream")

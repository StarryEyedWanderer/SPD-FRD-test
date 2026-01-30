import numpy as np

from spd_t4.metrics.change_point import event_time_from_radius


def test_event_detection():
    radius = np.concatenate([np.zeros(50), np.ones(50) * 2.0])
    event_time = event_time_from_radius(radius, window=10, threshold=1.0)
    assert event_time is not None
    assert event_time >= 10

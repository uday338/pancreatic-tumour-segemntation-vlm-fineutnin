import numpy as np
import pytest

SPACING_ZYX = (2.5, 0.8, 0.8)


def ball(shape, center, radius_vox):
    zz, yy, xx = np.indices(shape)
    c = np.asarray(center)
    r = np.asarray(radius_vox)
    return (((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2) <= 1


@pytest.fixture
def synthetic_case():
    """CT (HU), MSD-style label (1 pancreas, 2 tumor) and AMOS-style organ map."""
    rng = np.random.default_rng(0)
    shape = (24, 64, 64)
    ct = rng.normal(-80, 20, shape).astype(np.int16)  # fat-ish background
    pancreas = ball(shape, (12, 32, 32), (4, 8, 20))
    tumor = ball(shape, (12, 32, 48), (2, 4, 4))
    ct[pancreas] = rng.normal(60, 10, pancreas.sum())
    ct[tumor] = rng.normal(30, 15, tumor.sum())
    label = np.zeros(shape, np.uint8)
    label[pancreas] = 1
    label[tumor] = 2
    organs = np.zeros(shape, np.uint8)
    organs[pancreas] = 10
    duo = ball(shape, (12, 32, 55), (3, 5, 3)) & ~pancreas  # touches tumor side
    organs[duo] = 13
    organs[ball(shape, (12, 10, 10), (3, 4, 4))] = 6  # liver, far away
    return ct, label, organs

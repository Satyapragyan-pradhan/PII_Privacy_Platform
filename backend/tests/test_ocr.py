from ocr.preprocess import preprocess_image

import numpy as np


def test_preprocess():

    image = np.ones(
        (100, 100, 3),
        dtype=np.uint8
    ) * 255

    result = preprocess_image(
        image
    )

    assert result is not None
    assert len(result.shape) == 2
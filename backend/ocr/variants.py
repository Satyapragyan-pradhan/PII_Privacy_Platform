import cv2


def upscale(image, scale=3):
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


def grayscale(image):
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def adaptive_threshold(gray):
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )


def create_variants(image):
    """
    Create OCR variants for the complete image.

    Returns:
        dict[str, ndarray]
    """

    variants = {}

    # ---------------------------------------------------------
    # Original
    # ---------------------------------------------------------

    variants["original"] = image

    # ---------------------------------------------------------
    # Upscaled grayscale
    # ---------------------------------------------------------

    upscaled = upscale(
        image,
        scale=3
    )

    gray = grayscale(
        upscaled
    )

    variants["upscale_gray"] = gray

    # ---------------------------------------------------------
    # Adaptive threshold
    # ---------------------------------------------------------

    threshold = adaptive_threshold(
        gray
    )

    variants["upscale_adaptive_threshold"] = threshold

    # ---------------------------------------------------------
    # PSM 11 variant
    #
    # The image itself is the same. The engine will use
    # different Tesseract segmentation modes.
    # ---------------------------------------------------------

    variants["upscale_gray_psm11"] = gray

    return variants


def create_aadhaar_rois(image):
    """
    Create useful ROIs for an Aadhaar-like document.

    These coordinates are intentionally conservative and based
    on the current portrait-style Aadhaar image used during
    development.

    Returns:
        dict[str, ndarray]
    """

    height, width = image.shape[:2]

    rois = {}

    # ---------------------------------------------------------
    # Identity section
    #
    # Contains:
    #   Name
    #   DOB
    #   Gender
    # ---------------------------------------------------------

    y1 = int(height * 0.22)
    y2 = int(height * 0.67)

    x1 = int(width * 0.04)
    x2 = int(width * 0.68)

    identity = image[
        y1:y2,
        x1:x2
    ]

    if identity.size > 0:
        rois["identity"] = identity

    # ---------------------------------------------------------
    # Aadhaar number section
    # ---------------------------------------------------------

    y1 = int(height * 0.50)
    y2 = int(height * 0.99)

    x1 = int(width * 0.04)
    x2 = int(width * 0.68)

    number = image[
        y1:y2,
        x1:x2
    ]

    if number.size > 0:
        rois["number"] = number

    return rois


def create_roi_variants(image):
    """
    Generate variants specifically for ROI OCR.
    """

    variants = {}

    upscaled = upscale(
        image,
        scale=5
    )

    gray = grayscale(
        upscaled
    )

    variants["roi_upscale_gray"] = gray

    threshold = adaptive_threshold(
        gray
    )

    variants["roi_adaptive_threshold"] = threshold

    return variants
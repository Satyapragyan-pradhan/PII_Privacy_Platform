import cv2


def preprocess_image(image):
    """
    Legacy preprocessing function.

    Kept for compatibility with other parts of the project.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return gray
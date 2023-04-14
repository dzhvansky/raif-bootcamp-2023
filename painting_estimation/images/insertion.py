import cv2
import numpy as np

from painting_estimation.images.utils import image_size


def insert_img(
    img_to_insert: np.ndarray,
    background_img: np.ndarray,
    insertion_coef: float = 0.1,
    contour_to_insert: np.ndarray | None = None,
) -> np.ndarray:

    img_size = image_size(img_to_insert)
    bg_size = image_size(background_img)

    scale_coef = insertion_coef * bg_size.width / img_size.width

    img = cv2.resize(
        img_to_insert,
        dsize=tuple(map(round, (img_size.width * scale_coef, img_size.height * scale_coef))),
        interpolation=cv2.INTERPOLATION_AREA,
    )
    img_size = image_size(img)

    start_width = bg_size.width - img_size.width
    start_height = bg_size.height - img_size.height

    if contour_to_insert is not None:
        contour_to_insert = np.rint(contour_to_insert.astype(float) * scale_coef).astype(np.int32)
        bg_part = background_img[
            start_height : start_height + img_size.height, start_width : start_width + img_size.width
        ]
        cv2.drawContours(bg_part, [contour_to_insert], -1, (0, 0, 0), -1, cv2.LINE_AA)
        mask = np.zeros(bg_part.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour_to_insert], -1, (255, 255, 255), -1, cv2.LINE_AA)
        original_without_bg = cv2.bitwise_and(img, img, mask=mask)
        to_insert = cv2.bitwise_or(bg_part, original_without_bg)
    else:
        to_insert = img

    background_img[
        start_height : start_height + img_size.height, start_width : start_width + img_size.width
    ] = to_insert

    return background_img

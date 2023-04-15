import typing

import cv2
import numpy as np

from painting_estimation.images import utils
from painting_estimation.images.utils import ImgSize


def insert_image(
    img_to_insert: np.ndarray,
    background_img: np.ndarray,
    insertion_size_coef: float = 0.15,
    right_indent: float = 0.03,
    insertion_shape: typing.Literal["circle"] | None = None,
    contour_to_insert: np.ndarray | None = None,
) -> np.ndarray:
    dst_img: np.ndarray = background_img.copy()

    img_size: ImgSize = utils.image_size(img_to_insert)
    bg_size: ImgSize = utils.image_size(dst_img)

    scale_coef: float = insertion_size_coef * bg_size.width / min(img_size.width, img_size.height)

    img: np.ndarray = cv2.resize(
        img_to_insert,
        dsize=(round(img_size.width * scale_coef), round(img_size.height * scale_coef)),
        interpolation=cv2.INTER_AREA,
    )
    img_size = utils.image_size(img)

    indent: int = round(right_indent * min(bg_size.width, bg_size.height))
    start_width: int = bg_size.width - indent - img_size.width
    start_height: int = bg_size.height - indent - img_size.height

    if contour_to_insert is not None:
        contour_to_insert = np.rint(contour_to_insert.astype(float) * scale_coef).astype(np.int32)
    elif insertion_shape is not None:
        contour_to_insert = getattr(utils, f"build_{insertion_shape}_shape")(img)

    if contour_to_insert is not None:
        min_x, min_y = contour_to_insert.min(0).tolist()[0]
        max_x, max_y = contour_to_insert.max(0).tolist()[0]

        contour_to_insert -= (min_x, min_y)
        img = img[min_y : max_y + 1, min_x : max_x + 1]
        start_width += min_x + img_size.width - max_x - 1
        start_height += min_y + img_size.height - max_y - 1
        img_size = utils.image_size(img)

        bg_part: np.ndarray = dst_img[
            start_height : start_height + img_size.height, start_width : start_width + img_size.width
        ]
        cv2.drawContours(bg_part, [contour_to_insert], -1, (0, 0, 0), -1, cv2.LINE_AA)
        mask: np.ndarray = np.zeros(bg_part.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour_to_insert], -1, (255, 255, 255), -1, cv2.LINE_AA)
        original_without_bg: np.ndarray = cv2.bitwise_and(img, img, mask=mask)
        to_insert = cv2.bitwise_or(bg_part, original_without_bg)
    else:
        to_insert = img

    dst_img[start_height : start_height + img_size.height, start_width : start_width + img_size.width] = to_insert

    return dst_img

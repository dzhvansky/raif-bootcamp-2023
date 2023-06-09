import asyncio
import base64
import io
import logging
import pathlib
import random
import textwrap

import httpx
import numpy as np
import telegram
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from painting_estimation import models
from painting_estimation.images.insertion import insert_image
from painting_estimation.images.utils import cv2_image_from_byte_io, cv2_image_to_bytes
from painting_estimation.settings import settings


LOGGER: logging.Logger = logging.getLogger(__name__)
HTTP_CLIENT: httpx.AsyncClient = httpx.AsyncClient()
STYLE_TRANSFER_API: str = "https://aravinds1811-neural-style-transfer.hf.space/run/predict"
STYLE_TRANSFER_TIMEOUT: int = 180  # 3 minutes


DATA_DIR = pathlib.Path(__file__).parents[2] / "data" / "pics"
ARTIST_DIR = DATA_DIR / "artists"
PAINTING_DIR = DATA_DIR / "paintings"

ARTISTS: dict[str, str] = {
    "brjullov": "Карл Брюллов",
    "dali": "Сальвадор Дали",
    "davinci": "Леонардо Да Винчи",
    "malevich": "Казимир Малевич",
    "mone": "Клод Моне",
    "repin": "Илья Репин",
    "vrubel": "Михаил Врубель",
}


def random_artist_painting() -> tuple[str, bytes, bytes]:
    artists = list(ARTISTS.keys())
    random_idx = random.randint(0, len(artists) - 1)
    artist = artists[random_idx]
    artist_pic = (ARTIST_DIR / f"{artist}.jpg").read_bytes()
    painting_pic = (PAINTING_DIR / f"{artist}_pic.jpg").read_bytes()
    return ARTISTS[artist], artist_pic, painting_pic


async def download_to_memory(file: telegram.File) -> bytes:
    with io.BytesIO() as io_target:
        await file.download_to_memory(io_target)
        io_target.seek(0)
        return io_target.read()


async def fetch_price(image: bytes) -> models.Predict:
    LOGGER.info(f"Making prediction request to {settings.ml_api}")
    try:
        response: httpx.Response = await HTTP_CLIENT.post(settings.ml_api, files={"file": image})
    except httpx.ReadTimeout:
        LOGGER.error("Failed to fetch predictions from ML API, timeout")
        return models.Predict()
    if response.status_code != 200:
        LOGGER.error(f"Failed to fetch predictions from ML API, status_code: {response.status_code}")
    try:
        parsed_predict = models.Predict.parse_raw(response.content)
    except ValueError:
        LOGGER.error("Failed to parse API response, returning default value")
        return models.Predict()
    return parsed_predict


def base64_encode(data: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def base64_decode(data: str) -> bytes:
    return base64.b64decode(data)


async def style_transfer(content_file: telegram.File, style_file: telegram.File) -> bytes | None:
    raw_content: bytes
    raw_style: bytes
    raw_content, raw_style = await asyncio.gather(download_to_memory(content_file), download_to_memory(style_file))
    try:
        response: dict = httpx.post(
            STYLE_TRANSFER_API,
            json={"data": [base64_encode(raw_content), base64_encode(raw_style)]},
            timeout=STYLE_TRANSFER_TIMEOUT,
        ).json()
    except httpx.ReadTimeout:
        LOGGER.error("Failed to fetch style transfer result")
        return None
    return base64_decode(response["data"][0][22:])


async def artist_style_transfer(content_file: telegram.File, raw_style: bytes, raw_artist: bytes) -> bytes | None:
    raw_content: bytes
    raw_content = await download_to_memory(content_file)
    try:
        response: dict = httpx.post(
            STYLE_TRANSFER_API,
            json={"data": [base64_encode(raw_content), base64_encode(raw_style)]},
            timeout=STYLE_TRANSFER_TIMEOUT,
        ).json()
    except httpx.ReadTimeout:
        LOGGER.error("Failed to fetch style transfer result")
        return None
    styled_img: bytes = base64_decode(response["data"][0][22:])
    try:
        np_image = cv2_image_from_byte_io(io.BytesIO(styled_img))
        np_label: np.ndarray = cv2_image_from_byte_io(io.BytesIO(raw_artist))
        np_image = insert_image(np_label, np_image, insertion_shape="circle")
        byte_image: bytes = cv2_image_to_bytes(np_image)
    except Exception as exc:
        LOGGER.error(f"Unexpected Exception caught during image copyrighting: {exc}")
        return None
    return byte_image


async def label_adding(label_file: telegram.File, image_file: telegram.File) -> bytes | None:
    raw_label: bytes
    raw_image: bytes
    raw_label, raw_image = await asyncio.gather(download_to_memory(label_file), download_to_memory(image_file))
    try:
        np_label: np.ndarray = cv2_image_from_byte_io(io.BytesIO(raw_label))
        np_image: np.ndarray = cv2_image_from_byte_io(io.BytesIO(raw_image))
        np_image = insert_image(np_label, np_image, insertion_shape="circle")
        byte_image: bytes = cv2_image_to_bytes(np_image)
    except Exception as exc:
        LOGGER.error(f"Unexpected Exception caught during image copyrighting: {exc}")
        return None
    return byte_image


async def start(update: telegram.Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user_name: str = update.effective_user.first_name if update.effective_user else "коллега"
    LOGGER.info("Got /start command from {user_name}")
    if update.message:
        await update.message.reply_text(
            f"Привет, {user_name}! На связи ИИ. (Yes, I mean the ARTIFICIAL INTELLIGENCE).\n\n"
            "Я взял все картины проданные на аукционе sothebys.com и "
            "обучил свою нейросеть, чтобы предсказать их стоимость.\n\n"
            "Отправь мне фотографию своего рисунка и я скажу, how much bucks он может стоить.",
        )


async def estimate_price(update: telegram.Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    user: telegram.User | None = update.effective_user
    message: telegram.Message | None = update.message
    if message is None:
        LOGGER.error("Something strange happened, received empty message")
        return None

    latest_photo: telegram.PhotoSize = message.photo[-1]
    image: telegram.File = await latest_photo.get_file()
    prediction: models.Predict = await fetch_price(image=await download_to_memory(image))

    caption = textwrap.dedent(
        """
        Отличный piece of art! На черном рынке за него дадут {price:0.0f}$ ;)
    """
    ).format(price=prediction.price)

    LOGGER.info(f"Received photo to score from {user.full_name if user else 'somebody'}")
    await message.reply_text(caption)

    artist_name, artist_pic, style_pic = random_artist_painting()
    if artist_styled_image := await artist_style_transfer(image, style_pic, artist_pic):
        artist_styled_prediction: models.Predict = await fetch_price(artist_styled_image)
        await message.reply_photo(
            artist_styled_image,
            caption="Мало кто знает, но {artist} тоже вдохновлялся этим шедевром, ценник просто смешной - {price:0.0f}$".format(
                price=artist_styled_prediction.price,
                artist=artist_name,
            ),
        )
    else:
        artist_styled_prediction = models.Predict()

    if user:
        if user_photo := await user.get_profile_photos():
            latest_user_photo: telegram.PhotoSize = user_photo.photos[0][-1]
            if styled_photo := await style_transfer(await latest_user_photo.get_file(), image):
                styled_photo_price = max(prediction.price, artist_styled_prediction.price) + random.randint(2000, 5000)
                await message.reply_photo(
                    styled_photo,
                    caption="Ты униикальна(-ен) и неповторим(-а)! Но даже эта картина имеет цену {price:0.0f}$!".format(
                        price=styled_photo_price
                    ),
                )
            else:
                await message.reply_text(
                    "Хотели сделать кое-что интересное с твоей фотографией, но магия сломалась :( Приходи попозже!"
                )


APP = ApplicationBuilder().token(settings.telegram_token).build()
APP.add_handler(CommandHandler("start", start))
APP.add_handler(MessageHandler(filters.PHOTO, estimate_price))

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO, format="%(name)s :: %(levelname)s :: %(message)s'"
    )
    LOGGER.info("Launching our awesome bot!")
    APP.run_polling()

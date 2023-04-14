import asyncio
import base64
import io
import logging
import textwrap

import httpx
import telegram
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from painting_estimation import models
from painting_estimation.settings import settings


LOGGER: logging.Logger = logging.getLogger(__name__)
HTTP_CLIENT: httpx.AsyncClient = httpx.AsyncClient()
STYLE_TRANSFER_API: str = "https://aravinds1811-neural-style-transfer.hf.space/run/predict"
STYLE_TRANSFER_TIMEOUT: int = 30


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

    if user:
        if user_photo := await user.get_profile_photos():
            latest_user_photo: telegram.PhotoSize = user_photo.photos[0][-1]
            if styled_photo := await style_transfer(await latest_user_photo.get_file(), image):
                style_photo_prediction: models.Predict = await fetch_price(styled_photo)
                await message.reply_photo(
                    styled_photo,
                    caption="А ты хорош! За такую картину можно было бы выручить {price:0.0f}$!".format(
                        price=style_photo_prediction.price
                    ),
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

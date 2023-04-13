import io
import logging
import textwrap

import httpx
import numpy as np
import telegram
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from painting_estimation import models
from painting_estimation.settings import settings


LOGGER: logging.Logger = logging.getLogger(__name__)
HTTP_CLIENT: httpx.AsyncClient = httpx.AsyncClient()


async def fetch_price(image: telegram.File) -> models.Predict:
    with io.BytesIO() as io_target:
        await image.download_to_memory(io_target)
        io_target.seek(0)
        response: httpx.Response = await HTTP_CLIENT.post(settings.ml_api, files={"file": io_target.read()})
        if response.status_code != 200:
            LOGGER.error("Failed to fetch predictions from ML API")
    return models.Predict.parse_raw(response.content)


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
    prediction: models.Predict = await fetch_price(image=image)

    caption = textwrap.dedent(
        """
        Original Title: {title}
        Author: {author}
        Date: {date}
        Estimated price: {price} $ [Sothebys auction]
        Style: {style}
        Genre: {genre}
        Media: {media}
        Similar painting: {similar}
    """
    ).format(
        title="Unknown",
        author=user.full_name if user else "Unknown",
        date=np.random.choice(["2023", "Beginning of XXI century", "2020-es"]),
        price=prediction.price,
        style=np.random.choice(["Surrealism", "Realism", "Abstract Art", "Impressionism"]),
        genre=np.random.choice(
            ["animal painting", "portrait", "abstract", "illustration", "sketch and study", "figurative", "landscape"]
        ),
        media=np.random.choice(["oil", "pencil", "photo"]),
        similar="https://www.sothebys.com/en/buy/fine-art/paintings/abstract/_eve-ackroyd-woman-as-still-life-4eb9",
    )

    LOGGER.info(f"Received photo to score from {user.full_name if user else 'somebody'}")
    await message.reply_photo(latest_photo.file_id, caption=caption)


APP = ApplicationBuilder().token(settings.telegram_token).build()
APP.add_handler(CommandHandler("start", start))
APP.add_handler(MessageHandler(filters.PHOTO, estimate_price))

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO, format="%(name)s :: %(levelname)s :: %(message)s'"
    )
    LOGGER.info("Launching our awesome bot!")
    APP.run_polling()

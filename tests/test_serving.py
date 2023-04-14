from io import BytesIO


from painting_estimation.inference.serving import predict_painting_price


def test_predict(test_image: bytes) -> None:
    assert isinstance(predict_painting_price(BytesIO(test_image)), float)

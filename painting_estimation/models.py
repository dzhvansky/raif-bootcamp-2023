import pydantic
import pathlib
import joblib
from onnxruntime import InferenceSession
import numpy as np

MODELS_DIR = pathlib.Path(__file__).parent / "data/models/1/"

class Predict(pydantic.BaseModel):
    price: float = 2500



class FirstModel(pydantic.BaseModel):

    def reshape_image(self, image: np.array) -> np.array:
        width: int = image.shape[1]
        height: int = image.shape[0]
        if height != width:
            if height < width:
                image = np.pad(image, [(0, width - height), (0, 0), (0, 0)], mode='constant', constant_values=0)
            else: 
                image = np.pad(image, [(0, 0), (0, height - width), (0, 0)], mode='constant', constant_values=0)
        img_np = np.expand_dims(image.astype('float32'), 0)
        return img_np
        
    def get_features(self, image: np.array) ->  np.array: 
        """Get features from Inception model."""

        session = InferenceSession(str(MODELS_DIR / "incept_v3_1.onnx"))
        image = self.reshape_image(image)
        features = session.run(None, {'input0': image})
        return features[0]
    
    def predict(self, image: np.array) -> float:
        """Predict price using LightGBMRegressor."""

        features = self.get_features(image)
        lgbm = joblib.load(str(MODELS_DIR / "lgb.pkl"))
        price = np.expm1(lgbm.predict(features))[0] 
        price = price if price < 10000 else price * 1.5
        return price 



import pathlib
import typing

import numpy as np
import onnxruntime


class ModelProtocol(typing.Protocol):
    def __call__(self, nn_input: np.ndarray) -> np.ndarray:
        ...


class PreprocessorProtocol(typing.Protocol):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        ...


class PostprocessorProtocol(typing.Protocol):
    def __call__(self, nn_output: np.ndarray) -> np.ndarray | float:
        ...


class AggregatorProtocol(typing.Protocol):
    def __call__(self, outputs: typing.Mapping[str, np.ndarray]) -> float:
        ...


class ONNXModel:
    def __init__(self, model_path: typing.Union[pathlib.Path, str]):
        self.onnx_session = onnxruntime.InferenceSession(model_path)
        self._get_onnx_names()

    def _get_onnx_names(self) -> None:
        self.input_names = tuple(i.name for i in self.onnx_session.get_inputs())
        self.output_names = tuple(o.name for o in self.onnx_session.get_outputs())

    def __call__(self, nn_input: np.ndarray) -> np.ndarray:
        onnx_output, *_ = self.onnx_session.run(
            output_names=self.output_names, input_feed={self.input_names[0]: nn_input}
        )
        return onnx_output


class ModelServing:
    def __init__(
        self,
        model: ModelProtocol,
        preprocessor: PreprocessorProtocol,
        postprocessor: typing.Optional[PostprocessorProtocol] = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def __call__(self, image: np.ndarray) -> np.ndarray | float:
        nn_input: np.ndarray = self.preprocessor(image)
        # image feature set or final price estimation
        output: np.ndarray = self.model(nn_input)

        if self.postprocessor is not None:
            output = self.postprocessor(output)

        return output


class EnsembleServing:
    def __init__(self, models: typing.Mapping[str, ModelServing], aggregator: AggregatorProtocol):
        self.models = models
        self.aggregator = aggregator

    def __call__(self, image: np.ndarray, add_source_image: bool = False) -> float:
        outputs = {name: model(image) for name, model in self.models.items()}
        if add_source_image:
            outputs.update({"image": image})

        price: float = self.aggregator(outputs)
        return price

import pathlib
import typing

import numpy as np
import onnxruntime


class PreprocessorProtocol(typing.Protocol):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        ...


class PostprocessorProtocol(typing.Protocol):
    def __call__(self, onnx_output: np.ndarray) -> np.ndarray:
        ...


class AggregatorProtocol(typing.Protocol):
    def __call__(self, outputs: typing.Mapping[str, np.ndarray]) -> float:
        ...


class ModelServing:
    def __init__(
        self,
        model_path: typing.Union[pathlib.Path, str],
        preprocessor: PreprocessorProtocol,
        postprocessor: typing.Optional[PostprocessorProtocol] = None,
    ):
        self.preprocessor = preprocessor
        self.onnx_session = onnxruntime.InferenceSession(model_path)
        self.postprocessor = postprocessor

        self._get_onnx_names()

    def _get_onnx_names(self) -> None:
        self.input_names = tuple(i.name for i in self.onnx_session.get_inputs())
        self.output_names = tuple(o.name for o in self.onnx_session.get_outputs())

    def __call__(self, image: np.ndarray) -> np.ndarray:
        onnx_input: np.ndarray = self.preprocessor(image)
        onnx_output: np.ndarray

        # image feature set or final price estimation
        onnx_output, *_ = self.onnx_session.run(
            output_names=self.output_names, input_feed={self.input_names[0]: onnx_input}
        )

        if self.postprocessor is not None:
            onnx_output = self.postprocessor(onnx_output)

        return onnx_output


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

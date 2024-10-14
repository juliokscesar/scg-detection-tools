import numpy as np
from typing import Callable, List

class ImagePreprocess:
    def __init__(self, preproc_func: Callable[[np.ndarray], np.ndarray], *args, **kwargs):
        self._preproc_func = preproc_func
        self._args = args
        self._kwargs = kwargs

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.preprocess(img)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        return self._preproc_func(img, *self._args, **self._kwargs)

class ImagePreprocessPipeline:
    def __init__(self, preprocess_steps: List[ImagePreprocess]):
        self._steps = preprocess_steps

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        proc = img.copy()
        for preprocess_step in self._steps:
            proc = preprocess_step(proc)
        return proc


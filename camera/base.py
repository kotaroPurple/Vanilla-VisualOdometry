
from images.image_class import ImageWithFeature


class CameraBase:
    def __init__(self):
        pass

    def reset(self) -> None:
        NotImplementedError()

    def get_image(self) -> ImageWithFeature | None:
        NotImplementedError()

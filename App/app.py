import cv2
import numpy as np
from model import ConvNet, get_model, get_output_value, review
import pyttsx3


class App:
    def __init__(self, cam_width=240, cam_height=320, image_size=200):
        """"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        self.cap = cap
        self.size = image_size  # preferred image size
        self.start = False

        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        center_x = h // 2  # these are swapped due to some bug in opencv and numpy indexing
        center_y = w // 2
        self.start_x = int(center_x - self.size // 2)
        self.start_y = int(center_y - self.size // 2)
        # print(w, h, center_x, center_y, self.start_x, self.start_y, type(self.start_y))

        self.model = get_model()

        self.engine = pyttsx3.init()

        self.last_letter = True

    def main(self):
        """"""
        while True:
            success, img = self.cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = self.crop_to_center(img)
            # if not self.start:
            #     print(img.shape, type(img))
            #     self.start = True
            cv2.imshow("Image", img)
            img = review(img, self.size)
            output = self.model(img)
            output = get_output_value(output, self.last_letter)
            if output != self.last_letter:
                self.last_letter = output
                print(output)
                self.engine.say(output)
                self.engine.runAndWait()
            cv2.waitKey(1)

    def crop_to_center(self, image: np.ndarray):
        """
        Given image, crop to center based on preferred size
        :param image:
        :return:
        """
        # if not self.start:
        #     print(img.shape, self.size)
        return image[self.start_x:self.start_x + self.size, self.start_y:self.start_y + self.size]


if __name__ == '__main__':
    app = App()
    app.main()

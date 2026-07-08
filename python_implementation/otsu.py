import numpy as np


class OtsuThreshold:
    """
    Otsu automatic threshold with inverse binarisation.
    Very closed to cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inspired by https://en.wikipedia.org/wiki/Otsu%27s_method#MATLAB_implementation

    Default:
    - max val: 255
    - mode: BINARY_INV (255 if pixels <= thr else 0)

    Usage :
        thr, mask = OtsuThreshold().threshold(img_gray)
        thr, mask = OtsuThreshold.apply(img_gray)
    """

    def threshold(self, img: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Compute Otsu threshold.

        :param img: Grayscale image (H, W)
        :returns: Otsu threshold, binary image
        """
        img = self._check(img)
        thr = self._compute_otsu_threshold(img)
        mask = self._apply_binary_inv(img, thr)
        return thr, mask

    @staticmethod
    def apply(img: np.ndarray) -> tuple[int, np.ndarray]:
        """Static Shortcut"""
        return OtsuThreshold().threshold(img)

    # region Otsu threshold

    @staticmethod
    def _build_histogram(img: np.ndarray) -> list[int]:
        """Grayscale Histogram"""
        hist = [0] * 256
        rows, cols = img.shape
        for r in range(rows):
            for c in range(cols):
                hist[img[r][c]] += 1
        return hist

    @staticmethod
    def _compute_otsu_threshold(img: np.ndarray) -> int:
        """
        Maximise inter-class variance
        :return level: threshold
        """
        level = 0
        hist = OtsuThreshold._build_histogram(img)
        total = img.shape[0] * img.shape[1]
        top = 256
        sum_b = 0
        w_b = 0
        maximum = 0.0
        # sum1 = dot(0:255, hist)
        sum1 = 0
        for i in range(top):
            sum1 += i * hist[i]

        for ii in range(top):
            w_b += hist[ii]
            w_f = total - w_b
            sum_b += ii * hist[ii]

            if w_b > 0 and w_f > 0:
                m_f = (sum1 - sum_b) / w_f
                m_b = sum_b / w_b
                val = w_b * w_f * (m_b - m_f) * (m_b - m_f)

                if val >= maximum:
                    level = ii
                    maximum = val

        return level

    # endregion Otsu threshold

    # region Helpers

    @staticmethod
    def _check(img: np.ndarray) -> np.ndarray:
        """Check dimensions and types"""
        img = np.asarray(img)
        if img.ndim != 2:
            raise ValueError(f"Image must be grayscale (2D). Got shape: {img.shape}")

        # explicit 0-255
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return img

    @staticmethod
    def _apply_binary_inv(img: np.ndarray, thresh: int) -> np.ndarray:
        """
        255 if pixels <= thresh else 0
        flag THRESH_BINARY_INV in OpenCV
        """
        rows, cols = img.shape
        binary = np.empty((rows, cols), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                binary[r][c] = 255 if img[r][c] <= thresh else 0
        return binary

    # endregion Helpers
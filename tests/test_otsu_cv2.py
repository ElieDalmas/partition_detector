import numpy as np
import pytest
import cv2
from otsu import OtsuThreshold

class TestOtsuVsCv2:

    @pytest.mark.parametrize("fixture_name", ["gray_uniform", "gray_gradient"])
    def test_threshold_matches_cv2(self, fixture_name, request):
        img = request.getfixturevalue(fixture_name)
        cv_t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        my_t, _ = OtsuThreshold.apply(img)
        assert my_t == int(cv_t)

    @pytest.mark.parametrize("fixture_name", ["gray_uniform", "gray_gradient", "gray_bimodal"])
    def test_binary_matches_cv2(self, fixture_name, request):
        img = request.getfixturevalue(fixture_name)
        _, cv_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, my_bin = OtsuThreshold.apply(img)
        assert np.array_equal(cv_bin, my_bin)

from otsu import OtsuThreshold


class TestOtsuUnit:

    def test_threshold_in_range(self, gray_uniform):
        t, _ = OtsuThreshold.apply(gray_uniform)
        assert 0 <= t <= 255

    def test_binary_shape(self, gray_uniform):
        _, binary = OtsuThreshold.apply(gray_uniform)
        assert binary.shape == gray_uniform.shape

    def test_binary_only_0_255(self, gray_uniform):
        _, binary = OtsuThreshold.apply(gray_uniform)
        unique = set(binary.ravel().tolist())
        assert unique <= {0, 255}

    def test_bimodal_threshold(self, gray_bimodal):
        """On [20-80] U [160-230]"""
        t, _ = OtsuThreshold.apply(gray_bimodal)
        assert 80 <= t <= 160

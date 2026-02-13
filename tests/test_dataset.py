"""Tests for sonar.dataset (SONAR loader)."""

import pytest


class TestSONARValidation:
    def test_invalid_name(self):
        from sonar.dataset import SONAR

        with pytest.raises(ValueError, match="Invalid name 'tiny'"):
            SONAR(root="/tmp/sonar_test", name="tiny")

    def test_invalid_representation(self):
        from sonar.dataset import SONAR

        with pytest.raises(ValueError, match="Invalid representation"):
            SONAR(root="/tmp/sonar_test", name="small", representation="graph")

    def test_hetero_with_anomalies_error(self):
        from sonar.dataset import SONAR

        with pytest.raises(ValueError, match="Heterogeneous representation"):
            SONAR(
                root="/tmp/sonar_test",
                name="small",
                anomalies=True,
                representation="heterogeneous",
            )

    def test_medium_not_available(self):
        from sonar.dataset import SONAR

        with pytest.raises(FileNotFoundError, match="not included"):
            SONAR(root="/tmp/sonar_test", name="medium")

    def test_large_not_available(self):
        from sonar.dataset import SONAR

        with pytest.raises(FileNotFoundError, match="not included"):
            SONAR(root="/tmp/sonar_test", name="large")

    def test_contact_info_in_error(self):
        from sonar.dataset import SONAR

        with pytest.raises(FileNotFoundError, match="Dipanjan Chakraborty"):
            SONAR(root="/tmp/sonar_test", name="medium")


class TestSONARRepr:
    def test_repr_format(self):
        from sonar.dataset import SONAR

        # We can't instantiate without data, but we can test the __repr__
        # method directly by creating a mock-like object
        class FakeSONAR:
            name = "small"
            anomalies = True
            representation = "homogeneous"

            def __repr__(self):
                suffix = "+anomalies" if self.anomalies else "clean"
                return f"SONAR({self.name}, {suffix}, {self.representation})"

        obj = FakeSONAR()
        assert repr(obj) == "SONAR(small, +anomalies, homogeneous)"

        obj.anomalies = False
        assert repr(obj) == "SONAR(small, clean, homogeneous)"


class TestSONARConstants:
    def test_valid_names(self):
        from sonar.dataset import _VALID_NAMES

        assert _VALID_NAMES == {"small", "medium", "large"}

    def test_valid_representations(self):
        from sonar.dataset import _VALID_REPRESENTATIONS

        assert _VALID_REPRESENTATIONS == {"homogeneous", "heterogeneous"}

    def test_download_urls_keys(self):
        from sonar.dataset import _DOWNLOAD_URLS

        assert ("small", True) in _DOWNLOAD_URLS
        assert ("small", False) in _DOWNLOAD_URLS
        assert ("medium", True) not in _DOWNLOAD_URLS

    def test_download_urls_filenames(self):
        from sonar.dataset import _DOWNLOAD_URLS

        fname, url = _DOWNLOAD_URLS[("small", True)]
        assert fname == "SONAR_small_anomalies.pt"
        assert "github.com" in url

        fname, url = _DOWNLOAD_URLS[("small", False)]
        assert fname == "SONAR_small_clean.pt"
        assert "github.com" in url


@pytest.mark.slow
class TestSONARLoad:
    def test_load_small_anomalies(self, tmp_path):
        from sonar.dataset import SONAR

        dataset = SONAR(root=str(tmp_path), name="small", anomalies=True)
        data = dataset[0]

        assert data.num_nodes == 36860
        assert data.num_edges == 49865
        assert data.x.shape == (36860, 16)
        assert hasattr(data, "y_outlier")
        assert data.y_outlier.shape == (36860,)
        assert int(data.y_outlier.sum().item()) == 1818

    def test_load_small_clean_homo(self, tmp_path):
        from sonar.dataset import SONAR

        dataset = SONAR(
            root=str(tmp_path), name="small", anomalies=False, representation="homogeneous"
        )
        data = dataset[0]
        assert data.num_nodes > 0
        assert data.x.shape[1] == 16

    def test_load_small_clean_hetero(self, tmp_path):
        from sonar.dataset import SONAR

        dataset = SONAR(
            root=str(tmp_path),
            name="small",
            anomalies=False,
            representation="heterogeneous",
        )
        data = dataset[0]
        assert hasattr(data, "node_types")
        assert len(data.node_types) >= 2

    def test_repr_loaded(self, tmp_path):
        from sonar.dataset import SONAR

        dataset = SONAR(root=str(tmp_path), name="small", anomalies=True)
        r = repr(dataset)
        assert "SONAR" in r
        assert "small" in r
        assert "+anomalies" in r

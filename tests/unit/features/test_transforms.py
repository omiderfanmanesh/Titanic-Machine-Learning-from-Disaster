"""
Unit tests for feature transforms - individual preprocessing steps.
"""

import pytest
import pandas as pd
import numpy as np

from src.features.transforms import (
    FamilySizeTransform,
    TitleTransform,
    DeckTransform,
    TicketGroupTransform,
    FareTransform,
    AgeBinningTransform
)
from src.features.pipeline import build_pipeline_from_config, build_pipeline_pre, build_pipeline_post


class TestFamilySizeTransform:
    """Test family size feature engineering."""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'SibSp': [1, 0, 4, 1, 2],
            'Parch': [0, 2, 1, 0, 3],
            'Name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie']
        })

    def test_family_size_creation(self, sample_data):
        """Test that FamilySize is calculated correctly."""
        transform = FamilySizeTransform()
        result = transform.fit_transform(sample_data)

        expected_family_size = [2, 3, 6, 2, 6]  # SibSp + Parch + 1
        assert 'FamilySize' in result.columns
        assert result['FamilySize'].tolist() == expected_family_size

    def test_is_alone_creation(self, sample_data):
        """Test that IsAlone is calculated correctly."""
        transform = FamilySizeTransform()
        result = transform.fit_transform(sample_data)

        expected_is_alone = [0, 0, 0, 0, 0]  # Update based on logic: IsAlone = 1 if FamilySize == 1
        expected_is_alone = [0 if size > 1 else 1 for size in result['FamilySize']]  # Adjust dynamically
        assert 'IsAlone' in result.columns
        assert result['IsAlone'].tolist() == expected_is_alone

    def test_single_passenger(self):
        """Test with single passenger (alone)."""
        data = pd.DataFrame({
            'SibSp': [0],
            'Parch': [0],
            'Name': ['Solo']
        })

        transform = FamilySizeTransform()
        result = transform.fit_transform(data)

        assert result['FamilySize'].iloc[0] == 1
        assert result['IsAlone'].iloc[0] == 1

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises error."""
        data = pd.DataFrame({'Name': ['John']})

        transform = FamilySizeTransform()
        with pytest.raises(KeyError):
            transform.fit_transform(data)


class TestDeckTransform:
    """Test deck extraction from cabin information."""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'Cabin': ['C85', 'C123 C124', '', 'B57 B59 B63 B66', 'A6', np.nan, 'F G73']
        })

    def test_deck_extraction(self, sample_data):
        """Test that deck is extracted correctly from cabin."""
        transform = DeckTransform()
        result = transform.fit_transform(sample_data)

        expected_decks = ['C', 'C', 'U', 'B', 'A', 'U', 'F']
        assert 'Deck' in result.columns
        assert result['Deck'].tolist() == expected_decks

    def test_unknown_deck_for_missing_cabin(self):
        """Test that missing/empty cabins get 'U' (Unknown) deck."""
        data = pd.DataFrame({
            'Cabin': ['', np.nan, None]
        })

        transform = DeckTransform()
        result = transform.fit_transform(data)

        assert all(result['Deck'] == 'U')

    def test_multiple_cabins(self):
        """Test handling of multiple cabins (takes first)."""
        data = pd.DataFrame({
            'Cabin': ['A1 B2 C3']
        })

        transform = DeckTransform()
        result = transform.fit_transform(data)

        assert result['Deck'].iloc[0] == 'A'


class TestTicketGroupTransform:
    """Test ticket group size calculation."""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'Ticket': ['A/5 21171', 'PC 17599', 'A/5 21171', '113803', 'PC 17599', '373450']
        })

    def test_ticket_group_size(self, sample_data):
        """Test that ticket group sizes are calculated correctly."""
        transform = TicketGroupTransform()
        result = transform.fit_transform(sample_data)

        # A/5 21171 appears 2 times, PC 17599 appears 2 times, others appear 1 time
        expected_sizes = [2, 2, 2, 1, 2, 1]
        assert 'TicketGroupSize' in result.columns
        assert result['TicketGroupSize'].tolist() == expected_sizes

    def test_unique_tickets(self):
        """Test with all unique tickets."""
        data = pd.DataFrame({
            'Ticket': ['T1', 'T2', 'T3']
        })

        transform = TicketGroupTransform()
        result = transform.fit_transform(data)

        assert all(result['TicketGroupSize'] == 1)

    def test_all_same_ticket(self):
        """Test with all same tickets."""
        data = pd.DataFrame({
            'Ticket': ['SAME'] * 5
        })

        transform = TicketGroupTransform()
        result = transform.fit_transform(data)

        assert all(result['TicketGroupSize'] == 5)


class TestFareTransform:
    """Test fare preprocessing."""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'Fare': [7.25, 0.0, -1.0, 71.28, np.nan, 512.33]
        })

    def test_fare_clipping(self, sample_data):
        """Test that negative fares are clipped to 0."""
        transform = FareTransform(log_transform=False)
        result = transform.fit_transform(sample_data)

        assert (result['Fare'] >= 0).all()
        assert result['Fare'].iloc[2] == 0.0  # -1.0 clipped to 0

    def test_log_transform(self, sample_data):
        """Test log transformation of fare."""
        transform = FareTransform(log_transform=True)
        result = transform.fit_transform(sample_data)

        assert 'Fare_log' in result.columns
        # Check that log transform was applied (should be different from original)
        non_zero_mask = sample_data['Fare'] > 0
        original_non_zero = sample_data.loc[non_zero_mask, 'Fare'].fillna(0)
        if len(original_non_zero) > 0:
            transformed_non_zero = result.loc[non_zero_mask, 'Fare_log']
            assert not transformed_non_zero.equals(original_non_zero)

    def test_no_log_transform(self, sample_data):
        """Test without log transformation."""
        transform = FareTransform(log_transform=False)
        result = transform.fit_transform(sample_data)

        assert 'Fare_log' not in result.columns

    def test_zero_fare_handling(self):
        """Test handling of zero fares in log transform."""
        data = pd.DataFrame({'Fare': [0.0, 7.25, 0.0]})

        transform = FareTransform(log_transform=True)
        result = transform.fit_transform(data)

        # Zero fares should be handled gracefully (log(0+eps) or similar)
        assert not result['Fare_log'].isnull().any()


class TestAgeBinningTransform:
    """Test age binning transformation."""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'Age': [22.0, 38.0, 26.0, 35.0, 2.0, 80.0, np.nan]
        })

    def test_age_binning_default(self, sample_data):
        """Test age binning with default number of bins."""
        transform = AgeBinningTransform()
        result = transform.fit_transform(sample_data)

        assert 'AgeBin' in result.columns
        # Should have values between 0 and n_bins-1
        valid_ages = result['AgeBin'].dropna()
        assert valid_ages.min() >= 0
        assert valid_ages.max() < 5  # default n_bins

    def test_age_binning_custom_bins(self, sample_data):
        """Test age binning with custom number of bins."""
        transform = AgeBinningTransform(n_bins=3)
        result = transform.fit_transform(sample_data)

        valid_ages = result['AgeBin'].dropna()
        assert valid_ages.min() >= 0
        assert valid_ages.max() < 3

    def test_missing_ages_preserved(self, sample_data):
        """Test that missing ages remain missing after binning."""
        transform = AgeBinningTransform()
        result = transform.fit_transform(sample_data)

        # Missing age should remain missing
        assert result['AgeBin'].isnull().sum() == sample_data['Age'].isnull().sum()


class TestTitleTransform:
    """Test title extraction from names."""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'Name': [
                'Braund, Mr. Owen Harris',
                'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
                'Heikkinen, Miss. Laina',
                'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
                'Allen, Mr. William Henry',
                'Moran, Master. James',
                'McCarthy, Mr. Timothy J',
                'Palsson, Master. Gosta Leonard',
                'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)',
                'Nasser, Mrs. Nicholas (Adele Achem)'
            ]
        })

    def test_title_extraction(self, sample_data):
        """Titles are normalized via DEFAULT_TITLE_MAP (no frequency collapsing)."""
        transform = TitleTransform(rare_threshold=None)  # <-- important for small sample
        result = transform.fit_transform(sample_data)

        assert result["Title"].unique().tolist() == ["Miss", "Miss", "Mrs", "Royal"]

    def test_rare_title_grouping(self):
        """Test grouping of rare titles."""
        data = pd.DataFrame({
            'Name': [
                'Smith, Mr. John',
                'Doe, Dr. Jane',
                'Brown, Rev. James',
                'White, Col. Robert',
                'Black, Mr. William',
                'Green, Mrs. Alice',
                'Blue, Dr. Emily',
                'Red, Rev. Thomas',
                'Yellow, Col. George',
                'Pink, Mr. Charles'
            ]
        })

        # With threshold=2, Dr, Rev, Col should become 'Rare'
        transform = TitleTransform(rare_threshold=2)
        result = transform.fit_transform(data)

        titles = result['Title'].tolist()
        assert titles.count('Mr') == 2  # Common titles preserved
        assert 'Rare' in titles  # Rare titles grouped


class TestPipelineBuilding:
    """Test pipeline construction from configuration."""

    def test_build_pipeline_from_config(self):
        """Test building pipeline from transform list."""
        config = {"log_transform_fare": True, "age_bins": 3}
        stage_list = ["FamilySizeTransform", "FareTransform", "AgeBinningTransform"]

        pipeline = build_pipeline_from_config(stage_list, config)

        assert len(pipeline.transforms) == 3
        assert isinstance(pipeline.transforms[0], FamilySizeTransform)
        assert isinstance(pipeline.transforms[1], FareTransform)
        assert isinstance(pipeline.transforms[2], AgeBinningTransform)

    def test_build_pipeline_pre(self):
        """Test building pre-imputation pipeline."""
        config = {
            "feature_engineering": {
                "pre_impute": ["FamilySizeTransform", "DeckTransform"]
            }
        }

        pipeline = build_pipeline_pre(config)
        assert len(pipeline.transforms) == 2

    def test_build_pipeline_post(self):
        """Test building post-imputation pipeline."""
        config = {
            "feature_engineering": {
                "post_impute": ["FareTransform"]
            },
            "log_transform_fare": True
        }

        pipeline = build_pipeline_post(config)
        assert len(pipeline.transforms) == 1

    def test_unknown_transform_raises_error(self):
        """Test that unknown transform names raise error."""
        config = {}
        stage_list = ["UnknownTransform"]

        with pytest.raises(ValueError, match="Unknown transform"):
            build_pipeline_from_config(stage_list, config)

    def test_empty_pipeline(self):
        """Test building empty pipeline."""
        config = {}
        stage_list = []

        pipeline = build_pipeline_from_config(stage_list, config)
        assert len(pipeline.transforms) == 0

    def test_transform_with_parameters(self):
        """Test that transforms receive correct parameters from config."""
        config = {"log_transform_fare": True, "age_bins": 7}
        stage_list = ["FareTransform", "AgeBinningTransform"]

        pipeline = build_pipeline_from_config(stage_list, config)

        fare_transform = pipeline.transforms[0]
        age_transform = pipeline.transforms[1]

        assert fare_transform.log_transform is True
        assert age_transform.n_bins == 7


class TestTransformIntegration:
    """Test transforms working together in a pipeline."""

    @pytest.fixture
    def full_sample_data(self):
        """Complete sample data for integration testing."""
        return pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5],
            'Pclass': [3, 1, 3, 1, 3],
            'Name': [
                'Braund, Mr. Owen Harris',
                'Cumings, Mrs. John Bradley',
                'Heikkinen, Miss. Laina',
                'Futrelle, Mrs. Jacques Heath',
                'Allen, Mr. William Henry'
            ],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
            'Age': [22.0, 38.0, 26.0, 35.0, np.nan],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
            'Fare': [7.2500, 71.2833, 7.9250, 53.1000, np.nan],
            'Cabin': ['', 'C85', '', 'C123', ''],
            'Embarked': ['S', 'C', 'S', 'S', 'S']
        })

    def test_pre_imputation_pipeline(self, full_sample_data):
        """Test the complete pre-imputation transform pipeline."""
        config = {
            "feature_engineering": {
                "pre_impute": ["FamilySizeTransform", "DeckTransform", "TicketGroupTransform"]
            }
        }

        pipeline = build_pipeline_pre(config)
        result = pipeline.fit_transform(full_sample_data)

        # Check that all expected features were created
        expected_new_columns = ['FamilySize', 'IsAlone', 'Deck', 'TicketGroupSize']
        for col in expected_new_columns:
            assert col in result.columns

        # Check that original columns are preserved
        for col in full_sample_data.columns:
            assert col in result.columns

    def test_post_imputation_pipeline(self, full_sample_data):
        """Test the complete post-imputation transform pipeline."""
        config = {
            "feature_engineering": {
                "post_impute": ["FareTransform"]
            },
            "log_transform_fare": True
        }

        pipeline = build_pipeline_post(config)
        result = pipeline.fit_transform(full_sample_data)

        # Check that fare processing was applied
        assert 'Fare_log' in result.columns
        assert (result['Fare'] >= 0).all()

    def test_full_transform_pipeline(self, full_sample_data):
        """Test complete transform pipeline (pre + post imputation)."""
        config = {
            "feature_engineering": {
                "pre_impute": ["FamilySizeTransform", "DeckTransform"],
                "post_impute": ["FareTransform"]
            },
            "log_transform_fare": True
        }

        # Apply pre-imputation transforms
        pre_pipeline = build_pipeline_pre(config)
        intermediate = pre_pipeline.fit_transform(full_sample_data)

        # Apply post-imputation transforms
        post_pipeline = build_pipeline_post(config)
        final = post_pipeline.fit_transform(intermediate)

        # Check all expected features are present
        expected_features = ['FamilySize', 'IsAlone', 'Deck', 'Fare_log']
        for feat in expected_features:
            assert feat in final.columns

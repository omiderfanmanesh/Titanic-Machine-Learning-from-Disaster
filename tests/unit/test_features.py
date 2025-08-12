"""Tests for feature transformation components."""

import pytest
import pandas as pd
import numpy as np

from features.transforms import (
    FamilySizeTransform,
    TitleTransform,
    DeckTransform,
    TicketGroupTransform,
    FareTransform,
    AgeBinningTransform,
    MissingValueIndicatorTransform,
    FeaturePipeline
)


class TestFamilySizeTransform:
    """Test family size feature creation."""
    
    def test_family_size_transform(self, sample_titanic_data):
        """Test basic family size transformation."""
        transformer = FamilySizeTransform()
        
        # Fit and transform
        result = transformer.fit_transform(sample_titanic_data)
        
        # Check new columns exist
        assert "FamilySize" in result.columns
        assert "IsAlone" in result.columns
        
        # Check calculations are correct
        expected_family_size = sample_titanic_data["SibSp"] + sample_titanic_data["Parch"] + 1
        pd.testing.assert_series_equal(result["FamilySize"], expected_family_size)
        
        expected_is_alone = (expected_family_size == 1).astype(int)
        pd.testing.assert_series_equal(result["IsAlone"], expected_is_alone)
        
    def test_missing_columns_error(self):
        """Test error handling for missing columns."""
        transformer = FamilySizeTransform()
        
        df = pd.DataFrame({"col1": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            transformer.fit(df)


class TestTitleTransform:
    """Test title extraction transformation."""
    
    def test_title_extraction(self, sample_titanic_data):
        """Test title extraction from names."""
        transformer = TitleTransform(rare_threshold=2)
        
        result = transformer.fit_transform(sample_titanic_data)
        
        assert "Title" in result.columns
        
        # All should have "Mr" since we created names with "Mr."
        assert (result["Title"] == "Mr").all()
        
    def test_rare_title_handling(self):
        """Test rare title handling."""
        df = pd.DataFrame({
            "Name": [
                "Smith, Mr. John",
                "Jones, Mr. Bob", 
                "Brown, Dr. Alice",
                "Wilson, Rev. Joe"
            ]
        })
        
        transformer = TitleTransform(rare_threshold=2)
        result = transformer.fit_transform(df)
        
        # Mr should be kept (appears twice)
        # Dr and Rev should be mapped to "Rare"
        expected_titles = ["Mr", "Mr", "Rare", "Rare"]
        assert result["Title"].tolist() == expected_titles


class TestDeckTransform:
    """Test deck extraction transformation."""
    
    def test_deck_extraction(self):
        """Test deck extraction from cabin."""
        df = pd.DataFrame({
            "Cabin": ["A123", "B456", "C789", None, "X999", ""]
        })
        
        transformer = DeckTransform()
        result = transformer.fit_transform(df)
        
        assert "Deck" in result.columns
        
        expected = ["A", "B", "C", "U", "U", "U"]  # X and missing -> U
        assert result["Deck"].tolist() == expected


class TestTicketGroupTransform:
    """Test ticket group size transformation."""
    
    def test_ticket_group_transform(self):
        """Test ticket group size calculation."""
        df = pd.DataFrame({
            "Ticket": ["TICKET1", "TICKET1", "TICKET2", "TICKET3"]
        })
        
        transformer = TicketGroupTransform()
        result = transformer.fit_transform(df)
        
        assert "TicketGroupSize" in result.columns
        
        expected = [2, 2, 1, 1]  # TICKET1 appears twice
        assert result["TicketGroupSize"].tolist() == expected
        
    def test_unseen_ticket_handling(self):
        """Test handling of unseen tickets in transform."""
        train_df = pd.DataFrame({
            "Ticket": ["TICKET1", "TICKET1", "TICKET2"]
        })
        
        test_df = pd.DataFrame({
            "Ticket": ["TICKET1", "TICKET3"]  # TICKET3 is new
        })
        
        transformer = TicketGroupTransform()
        transformer.fit(train_df)
        result = transformer.transform(test_df)
        
        expected = [2, 1]  # TICKET1 has size 2, TICKET3 defaults to 1
        assert result["TicketGroupSize"].tolist() == expected


class TestFareTransform:
    """Test fare transformation."""
    
    def test_fare_imputation(self):
        """Test fare missing value imputation."""
        df = pd.DataFrame({
            "Fare": [10.0, 20.0, np.nan, 30.0, np.nan]
        })
        
        transformer = FareTransform()
        result = transformer.fit_transform(df)
        
        # Missing values should be filled with median (20.0)
        expected = [10.0, 20.0, 20.0, 30.0, 20.0]
        np.testing.assert_array_equal(result["Fare"].values, expected)
        
    def test_log_transformation(self):
        """Test log transformation of fare."""
        df = pd.DataFrame({
            "Fare": [1.0, 2.0, 3.0]
        })
        
        transformer = FareTransform(log_transform=True)
        result = transformer.fit_transform(df)
        
        assert "Fare_log" in result.columns
        
        expected_log = np.log1p([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result["Fare_log"].values, expected_log)


class TestAgeBinningTransform:
    """Test age binning transformation."""
    
    def test_age_binning(self):
        """Test age binning functionality."""
        df = pd.DataFrame({
            "Age": [5, 15, 25, 35, 45, 55, 65, 75]
        })
        
        transformer = AgeBinningTransform(n_bins=4)
        result = transformer.fit_transform(df)
        
        assert "AgeBin" in result.columns
        
        # Should have 4 unique bins
        assert result["AgeBin"].nunique() <= 4


class TestMissingValueIndicatorTransform:
    """Test missing value indicator creation."""
    
    def test_missing_indicators(self):
        """Test missing value indicator creation."""
        df = pd.DataFrame({
            "A": [1, 2, np.nan, 4],
            "B": [1, 2, 3, 4],  # No missing
            "C": [np.nan, np.nan, 3, 4]  # Many missing
        })
        
        transformer = MissingValueIndicatorTransform(missing_threshold=0.1)
        result = transformer.fit_transform(df)
        
        # Should create indicators for A and C (both have >10% missing)
        assert "A_missing" in result.columns
        assert "C_missing" in result.columns
        assert "B_missing" not in result.columns  # B has no missing values
        
        # Check indicator values
        expected_a = [0, 0, 1, 0]
        expected_c = [1, 1, 0, 0]
        
        assert result["A_missing"].tolist() == expected_a
        assert result["C_missing"].tolist() == expected_c


class TestFeaturePipeline:
    """Test feature pipeline composition."""
    
    def test_feature_pipeline(self, sample_titanic_data):
        """Test chaining multiple transformations."""
        transforms = [
            FamilySizeTransform(),
            TitleTransform(),
            DeckTransform()
        ]
        
        pipeline = FeaturePipeline(transforms)
        result = pipeline.fit_transform(sample_titanic_data)
        
        # Should have all new features
        assert "FamilySize" in result.columns
        assert "IsAlone" in result.columns
        assert "Title" in result.columns
        assert "Deck" in result.columns
        
    def test_pipeline_fit_transform_consistency(self, sample_titanic_data):
        """Test that pipeline.fit_transform == pipeline.fit().transform()."""
        transforms = [FamilySizeTransform(), TitleTransform()]
        
        pipeline1 = FeaturePipeline(transforms)
        result1 = pipeline1.fit_transform(sample_titanic_data)
        
        pipeline2 = FeaturePipeline(transforms)
        pipeline2.fit(sample_titanic_data)
        result2 = pipeline2.transform(sample_titanic_data)
        
        pd.testing.assert_frame_equal(result1, result2)

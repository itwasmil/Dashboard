import pytest
import shap
from dashboard import load_data, welcome_page_wrapper, welcome_page




from unittest.mock import MagicMock


import pandas as pd
import os
from unittest.mock import patch, mock_open

MOCK_DATA_PATH = 'mock_data.csv'
MOCK_RAW_DATA_PATH = 'mock_raw_data.csv'
MOCK_MERGED = 'mock_merged.csv'
MOCK_THRESHOLD_PATH = 'mock_threshold.txt'

@pytest.fixture
def mock_open_file(monkeypatch):
    # Mock the open function to return a file with mock content
    mock_file_content = '42.0'
    monkeypatch.setattr('builtins.open', mock_open(read_data=mock_file_content).return_value)

def test_load_data(mock_open_file, monkeypatch):
    # Mock the file reading functions
    with patch('pandas.read_csv') as mock_read_csv:
        # Set up mock return values for pandas read_csv
        mock_read_csv.side_effect = lambda x: pd.DataFrame({'col1': [1, 2, 3]})

        # Call the function under test
        result = load_data()

        # Assertions
        assert len(result) == 4
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DataFrame)
        assert isinstance(result[2], pd.DataFrame)
        assert isinstance(result[3], float)


def test_welcome_page(mock_beta_columns):
    # Mocking Streamlit functions
    st = MagicMock()

    # Call the wrapper function under test
    welcome_page_wrapper(st)

def welcome_page_wrapper(st):
    return welcome_page(st)

# Mocking the st.beta_columns method
@patch('dashboard.st.beta_columns', return_value=[MagicMock(), MagicMock(), MagicMock()])
def welcome_page_wrapper(mock_beta_columns):
    return welcome_page(st)
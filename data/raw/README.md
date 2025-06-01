# Raw Data Directory

This directory contains the raw, unprocessed data files used by the application.

## Files
- `content.csv`: Raw content data
- `engagements.csv`: Raw engagement data
- `students.csv`: Raw student data

## Data Formats
- All files are in CSV format
- UTF-8 encoding
- First row contains headers
- No data transformation has been applied

## Usage
These files should not be modified directly. Instead:
1. Use the ingestion scripts in `scripts/` to load data
2. Use the data processing modules in `data/processing/` to transform data
3. Processed data will be stored in `data/processed/` 
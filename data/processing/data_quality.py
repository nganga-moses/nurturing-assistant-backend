import pandas as pd
from typing import Dict, Any

class DataValidator:
    def __init__(self):
        self.expected_types = {
            'student_id': str,
            'gpa': float,
            'sat_score': float,
            'act_score': float,
            'application_status': str,
            'funnel_stage': str,
            'interaction_count': int,
            'application_likelihood_score': float,
            'dropout_risk_score': float
        }
        self.expected_ranges = {
            'gpa': (0.0, 4.0),
            'sat_score': (400, 1600),
            'act_score': (1, 36),
            'interaction_count': (0, float('inf')),
            'application_likelihood_score': (0.0, 1.0),
            'dropout_risk_score': (0.0, 1.0),
            'age': (10, 100)
        }

    def validate_data(self, data: pd.DataFrame) -> Dict[str, any]:
        """Validate data quality and return test-expected keys."""
        errors = []
        warnings = []
        # Type errors
        type_errors = self._validate_types(data)
        for col, count in type_errors.items():
            if count > 0:
                errors.append({'type': 'type_error', 'message': f"{count} type errors in {col}", 'location': col})
        # Range violations
        range_violations = self._validate_ranges(data)
        for col, count in range_violations.items():
            if count > 0:
                errors.append({'type': 'range_violation', 'message': f"{count} range violations in {col}", 'location': col})
        # Missing values
        missing_values = self._validate_missing(data)
        for col, count in missing_values.items():
            if count > 0:
                warnings.append({'type': 'missing_value', 'message': f"{count} missing values in {col}", 'location': col})
        # Completeness
        completeness = self._validate_completeness(data)
        is_valid = len(errors) == 0
        return {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'type_errors': type_errors,
            'range_violations': range_violations,
            'missing_values': missing_values,
            'completeness': completeness
        }

    def _validate_types(self, data: pd.DataFrame) -> Dict[str, int]:
        """Validate data types."""
        type_errors = {}
        for column, expected_type in self.expected_types.items():
            if column not in data.columns:
                continue
            type_errors[column] = sum(not isinstance(x, expected_type) for x in data[column])
        return type_errors

    def _validate_ranges(self, data: pd.DataFrame) -> Dict[str, int]:
        """Validate value ranges."""
        range_violations = {}
        for column, (min_val, max_val) in self.expected_ranges.items():
            if column not in data.columns:
                continue
            range_violations[column] = sum((data[column] < min_val) | (data[column] > max_val))
        return range_violations

    def _validate_missing(self, data: pd.DataFrame) -> Dict[str, int]:
        """Validate missing values."""
        missing_values = {}
        for column in self.expected_types.keys():
            if column not in data.columns:
                continue
            missing_values[column] = data[column].isna().sum()
        return missing_values

    def _validate_completeness(self, data: pd.DataFrame) -> float:
        """Calculate data completeness."""
        total_cells = len(data) * len(self.expected_types)
        missing_cells = sum(self._validate_missing(data).values())
        return 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0 
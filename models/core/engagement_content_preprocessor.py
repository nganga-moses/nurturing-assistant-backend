import pandas as pd
import numpy as np
from typing import Dict, Any

class EngagementContentPreprocessor:
    """
    Prepares data for the engagement + content recommendation model.
    Joins engagement and content data, handles missing content, and creates features for each tower.
    """
    def __init__(self):
        pass

    def preprocess(self, students_df: pd.DataFrame, engagements_df: pd.DataFrame, content_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Joins engagement and content data, handles missing content, and prepares features.
        Args:
            students_df: DataFrame of student profiles
            engagements_df: DataFrame of engagement history
            content_df: DataFrame of content items
        Returns:
            Dict with processed DataFrames and feature arrays
        """
        # Join engagement and content data (left join, so missing content is allowed)
        merged = engagements_df.merge(
            content_df,
            how='left',
            left_on='engagement_content_id',
            right_on='content_id',
            suffixes=('', '_content')
        )

        # Fill missing content fields with defaults
        merged['content_id'] = merged['content_id'].fillna('NO_CONTENT')
        merged['content_type'] = merged.get('content_type', pd.Series(['none']*len(merged)))
        merged['content_body'] = merged.get('content_body', pd.Series(['']*len(merged)))
        merged['content_meta'] = merged.get('content_meta', pd.Series([{}]*len(merged)))

        # Prepare student features (can be extended as needed)
        student_features = students_df.set_index('student_id').to_dict(orient='index')

        # Prepare engagement type features (categorical)
        engagement_types = merged['engagement_type'].unique().tolist()
        engagement_type_to_idx = {et: i for i, et in enumerate(engagement_types)}
        merged['engagement_type_idx'] = merged['engagement_type'].map(engagement_type_to_idx)

        # Prepare content features (text/meta, handle missing)
        content_ids = merged['content_id'].unique().tolist()
        content_id_to_idx = {cid: i for i, cid in enumerate(content_ids)}
        merged['content_id_idx'] = merged['content_id'].map(content_id_to_idx)

        # Mask for missing content
        merged['has_content'] = merged['content_id'] != 'NO_CONTENT'

        # Prepare output
        return {
            'students_df': students_df,
            'engagements_df': engagements_df,
            'content_df': content_df,
            'merged_df': merged,
            'student_features': student_features,
            'engagement_types': engagement_types,
            'engagement_type_to_idx': engagement_type_to_idx,
            'content_ids': content_ids,
            'content_id_to_idx': content_id_to_idx
        } 
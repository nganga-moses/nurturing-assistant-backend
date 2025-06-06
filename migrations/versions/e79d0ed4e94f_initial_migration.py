"""initial migration

Revision ID: e79d0ed4e94f
Revises: 
Create Date: 2025-06-03 11:17:32.702114

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e79d0ed4e94f'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('custom_fields',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('applies_to', sa.String(), nullable=False),
    sa.Column('field_type', sa.String(), nullable=False),
    sa.Column('is_required', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('engagement_content',
    sa.Column('content_id', sa.String(), nullable=False),
    sa.Column('engagement_type', sa.String(), nullable=True),
    sa.Column('content_category', sa.String(), nullable=True),
    sa.Column('content_description', sa.String(), nullable=True),
    sa.Column('content_features', sa.JSON(), nullable=True),
    sa.Column('success_rate', sa.Float(), nullable=True),
    sa.Column('target_funnel_stage', sa.String(), nullable=True),
    sa.Column('appropriate_for_risk_level', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('content_id')
    )
    op.create_table('error_logs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('error_type', sa.String(), nullable=False),
    sa.Column('details', sa.String(), nullable=False),
    sa.Column('row_number', sa.Integer(), nullable=True),
    sa.Column('file_name', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('resolved', sa.Boolean(), nullable=True),
    sa.Column('resolved_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('integration_configs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('type', sa.String(), nullable=False),
    sa.Column('config', sa.JSON(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_table('recommendation_feedback_metrics',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('recommendation_type', sa.String(), nullable=True),
    sa.Column('total_shown', sa.Integer(), nullable=True),
    sa.Column('acted_count', sa.Integer(), nullable=True),
    sa.Column('ignored_count', sa.Integer(), nullable=True),
    sa.Column('untouched_count', sa.Integer(), nullable=True),
    sa.Column('avg_time_to_action', sa.Float(), nullable=True),
    sa.Column('avg_time_to_completion', sa.Float(), nullable=True),
    sa.Column('completion_rate', sa.Float(), nullable=True),
    sa.Column('satisfaction_score', sa.Float(), nullable=True),
    sa.Column('dropoff_rates', sa.JSON(), nullable=True),
    sa.Column('last_updated', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('recommendation_settings',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('mode', sa.String(), nullable=True),
    sa.Column('schedule_type', sa.String(), nullable=True),
    sa.Column('schedule_day', sa.String(), nullable=True),
    sa.Column('schedule_time', sa.String(), nullable=True),
    sa.Column('batch_size', sa.Integer(), nullable=True),
    sa.Column('last_run', sa.DateTime(), nullable=True),
    sa.Column('next_run', sa.DateTime(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('roles',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_table('student_profiles',
    sa.Column('student_id', sa.String(), nullable=False),
    sa.Column('demographic_features', sa.JSON(), nullable=True),
    sa.Column('application_status', sa.String(), nullable=True),
    sa.Column('funnel_stage', sa.String(), nullable=True),
    sa.Column('first_interaction_date', sa.DateTime(), nullable=True),
    sa.Column('last_interaction_date', sa.DateTime(), nullable=True),
    sa.Column('interaction_count', sa.Integer(), nullable=True),
    sa.Column('application_likelihood_score', sa.Float(), nullable=True),
    sa.Column('dropout_risk_score', sa.Float(), nullable=True),
    sa.Column('last_recommended_engagement_id', sa.String(), nullable=True),
    sa.Column('last_recommended_engagement_date', sa.DateTime(), nullable=True),
    sa.Column('last_recommendation_at', sa.DateTime(), nullable=True),
    sa.Column('enrollment_agent_id', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('gpa', sa.Float(), nullable=True),
    sa.Column('sat_score', sa.Float(), nullable=True),
    sa.Column('act_score', sa.Float(), nullable=True),
    sa.PrimaryKeyConstraint('student_id')
    )
    op.create_table('engagement_history',
    sa.Column('engagement_id', sa.String(), nullable=False),
    sa.Column('student_id', sa.String(), nullable=True),
    sa.Column('engagement_type', sa.String(), nullable=True),
    sa.Column('engagement_content_id', sa.String(), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.Column('engagement_response', sa.String(), nullable=True),
    sa.Column('engagement_metrics', sa.JSON(), nullable=True),
    sa.Column('funnel_stage_before', sa.String(), nullable=True),
    sa.Column('funnel_stage_after', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['engagement_content_id'], ['engagement_content.content_id'], ),
    sa.ForeignKeyConstraint(['student_id'], ['student_profiles.student_id'], ),
    sa.PrimaryKeyConstraint('engagement_id')
    )
    op.create_table('engagement_types',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('required_fields', sa.JSON(), nullable=True),
    sa.Column('integration_id', sa.Integer(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['integration_id'], ['integration_configs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_table('status_changes',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('student_id', sa.String(), nullable=True),
    sa.Column('field', sa.String(), nullable=True),
    sa.Column('old_value', sa.String(), nullable=True),
    sa.Column('new_value', sa.String(), nullable=True),
    sa.Column('batch_id', sa.String(), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['student_id'], ['student_profiles.student_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('stored_recommendations',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('student_id', sa.String(), nullable=True),
    sa.Column('recommendations', sa.JSON(), nullable=True),
    sa.Column('generated_at', sa.DateTime(), nullable=True),
    sa.Column('expires_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['student_id'], ['student_profiles.student_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('supabase_id', sa.String(), nullable=False),
    sa.Column('username', sa.String(), nullable=False),
    sa.Column('email', sa.String(), nullable=False),
    sa.Column('role', sa.String(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('first_name', sa.String(), nullable=True),
    sa.Column('last_name', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['role'], ['roles.name'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('supabase_id'),
    sa.UniqueConstraint('username')
    )
    op.create_table('recommendation_actions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('student_id', sa.String(), nullable=True),
    sa.Column('recommendation_id', sa.Integer(), nullable=True),
    sa.Column('action_type', sa.String(), nullable=True),
    sa.Column('action_timestamp', sa.DateTime(), nullable=True),
    sa.Column('time_to_action', sa.Integer(), nullable=True),
    sa.Column('action_completed', sa.Integer(), nullable=True),
    sa.Column('dropoff_point', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['recommendation_id'], ['stored_recommendations.id'], ),
    sa.ForeignKeyConstraint(['student_id'], ['student_profiles.student_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('settings',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('key', sa.String(), nullable=False),
    sa.Column('value', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('settings')
    op.drop_table('recommendation_actions')
    op.drop_table('users')
    op.drop_table('stored_recommendations')
    op.drop_table('status_changes')
    op.drop_table('engagement_types')
    op.drop_table('engagement_history')
    op.drop_table('student_profiles')
    op.drop_table('roles')
    op.drop_table('recommendation_settings')
    op.drop_table('recommendation_feedback_metrics')
    op.drop_table('integration_configs')
    op.drop_table('error_logs')
    op.drop_table('engagement_content')
    op.drop_table('custom_fields')
    # ### end Alembic commands ###

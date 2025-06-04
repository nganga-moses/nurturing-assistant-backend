# Remove: from .base import Base
# If you need Base, use:
from .student_profile import StudentProfile
from .stored_recommendation import StoredRecommendation
from .recommendation_action import RecommendationAction
from .recommendation_feedback_metrics import RecommendationFeedbackMetrics
from .recommendation_settings import RecommendationSettings
from .engagement_content import EngagementContent
from .engagement_history import EngagementHistory
from .engagement_type import EngagementType
from .custom_field import CustomField
from .status_change import StatusChange
from .error_log import ErrorLog
from .integration_config import IntegrationConfig
from .settings import Settings
from .user import User, Role
from .recommendation import Recommendation
from database.base import Base

__all__ = [
    'StudentProfile',
    'StoredRecommendation',
    'RecommendationAction',
    'RecommendationFeedbackMetrics',
    'RecommendationSettings',
    'EngagementContent',
    'EngagementHistory',
    'EngagementType',
    'CustomField',
    'StatusChange',
    'ErrorLog',
    'IntegrationConfig',
    'Settings',
    'User',
    'Role',
    'Recommendation'
]

from fastapi import APIRouter

api_router = APIRouter()

from api.routes.students import router as students_router
from api.routes.recommendations import router as recommendations_router
from api.routes.dashboard import router as dashboard_router
from api.routes.likelihood import router as likelihood_router
from api.routes.risk import router as risk_router
from api.routes.bulk_actions import router as bulk_actions_router
from api.routes.recommendation_settings import router as recommendation_settings_router
from api.routes.reports import router as reports_router
from api.routes.vp import router as vp_router
from api.routes.notifications import router as notifications_router
from api.routes.custom_fields import router as custom_fields_router
from api.routes.integrations import router as integrations_router
from api.routes.engagement_types import router as engagement_types_router
from api.routes.settings import router as settings_router
from api.routes.users import router as users_router
from api.routes.initial_setup import router as initial_setup_router

api_router.include_router(students_router, tags=["Students"])
api_router.include_router(recommendations_router, tags=["Recommendations"])
api_router.include_router(dashboard_router, tags=["Dashboard"])
api_router.include_router(likelihood_router, tags=["Likelihood"])
api_router.include_router(risk_router, tags=["Risk"])
api_router.include_router(bulk_actions_router, tags=["Bulk Actions"])
api_router.include_router(recommendation_settings_router, tags=["Recommendation Settings"])
api_router.include_router(reports_router, tags=["Reports"])
api_router.include_router(vp_router, tags=["VP"])
api_router.include_router(notifications_router, tags=["Notifications"])
api_router.include_router(custom_fields_router, tags=["Custom Fields"])
api_router.include_router(integrations_router, tags=["Integrations"])
api_router.include_router(engagement_types_router, tags=["Engagement Types"])
api_router.include_router(settings_router, tags=["Settings"])
api_router.include_router(users_router, tags=["Users"])
api_router.include_router(initial_setup_router, tags=["Initial Setup"])

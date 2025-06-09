from fastapi import APIRouter

api_router = APIRouter()

# Import all your original routes
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
from api.routes.success_factors import router as success_factors_router

# Import the new model router
from api.routes.model import router as model_router

# Include all original routes
api_router.include_router(students_router, tags=["students"])
api_router.include_router(recommendations_router, tags=["recommendations"])
api_router.include_router(dashboard_router, tags=["dashboard"])
api_router.include_router(likelihood_router, tags=["likelihood"])
api_router.include_router(risk_router, tags=["risk"])
api_router.include_router(bulk_actions_router, tags=["bulk-actions"])
api_router.include_router(recommendation_settings_router, tags=["recommendation-settings"])
api_router.include_router(reports_router, tags=["reports"])
api_router.include_router(vp_router, tags=["vp"])
api_router.include_router(notifications_router, tags=["notifications"])
api_router.include_router(custom_fields_router, tags=["custom-fields"])
api_router.include_router(integrations_router, tags=["integrations"])
api_router.include_router(engagement_types_router, tags=["engagement-types"])
api_router.include_router(settings_router, tags=["settings"])
api_router.include_router(users_router, tags=["users"])
api_router.include_router(initial_setup_router, tags=["initial-setup"])
api_router.include_router(success_factors_router, tags=["success-factors"])

# Include the new model routes with prefix
api_router.include_router(model_router, prefix="/api/v1/model", tags=["AI-Model"])

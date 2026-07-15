from fastapi import FastAPI

from bimba3d_backend.app.api.training_pipeline_streaming import router as exif_router


def test_exif_streaming_routes_are_available_under_workflow_prefix():
    app = FastAPI()
    app.include_router(exif_router, prefix="/api/workflow/pipelines")

    paths = {route.path for route in app.routes}

    assert "/api/workflow/pipelines/{pipeline_id}/test-exif/start" in paths
    assert "/api/workflow/pipelines/{pipeline_id}/test-exif/progress/{task_id}" in paths
    assert "/api/workflow/pipelines/{pipeline_id}/test-exif/results" in paths
    assert "/api/workflow/pipelines/{pipeline_id}/test-exif/stop/{task_id}" in paths

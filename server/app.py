from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from .routes import predict_route

app = FastAPI()

app.include_router(predict_route)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422, content={"error": str(exc.errors()[0]["ctx"]["error"])}
    )


@app.exception_handler(Exception)
async def validation_exception_handler(request: Request, exc):
    return {"detail": str(exc)}

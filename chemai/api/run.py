"""Local entry point:  python -m chemai.api.run"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("chemai.api.app:app", host="0.0.0.0", port=8000, reload=True)

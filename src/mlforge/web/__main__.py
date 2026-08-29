"""Run the local MLForge web API."""

import uvicorn


def main() -> None:
    """Start the single-user API on the loopback interface."""
    uvicorn.run(
        "mlforge.web.app:create_app",
        factory=True,
        host="127.0.0.1",
        port=8000,
    )


if __name__ == "__main__":
    main()

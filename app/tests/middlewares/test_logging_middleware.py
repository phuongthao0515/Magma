from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request
from starlette.responses import Response

from src.middleware.logging_middleware import LoggingMiddleware


class TestLoggingMiddleware:
    @pytest.mark.asyncio
    async def test_dispatch_sets_request_id_and_response_header(self, monkeypatch):
        middleware = LoggingMiddleware(app=AsyncMock())
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/health",
                "headers": [],
                "query_string": b"",
                "client": ("127.0.0.1", 1234),
                "server": ("testserver", 80),
                "scheme": "http",
                "http_version": "1.1",
            }
        )
        call_next = AsyncMock(return_value=Response(status_code=204))

        monkeypatch.setattr("src.middleware.logging_middleware.uuid.uuid4", lambda: "fixed-request-id")

        response = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once_with(request)
        assert request.state.request_id == "fixed-request-id"
        assert response.headers["X-Request-Id"] == "fixed-request-id"

    @pytest.mark.asyncio
    async def test_dispatch_logs_request_and_response(self, monkeypatch):
        middleware = LoggingMiddleware(app=AsyncMock())
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/v1/tasks",
                "headers": [],
                "query_string": b"",
                "client": ("127.0.0.1", 1234),
                "server": ("testserver", 80),
                "scheme": "http",
                "http_version": "1.1",
            }
        )
        call_next = AsyncMock(return_value=Response(status_code=201))
        info_messages = []
        perf_counter_values = iter([10.0, 10.25])

        monkeypatch.setattr("src.middleware.logging_middleware.uuid.uuid4", lambda: "log-request-id")
        monkeypatch.setattr(
            "src.middleware.logging_middleware.time.perf_counter",
            lambda: next(perf_counter_values),
        )
        monkeypatch.setattr(
            "src.middleware.logging_middleware.logger.info",
            lambda message: info_messages.append(message),
        )

        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 201
        assert info_messages == [
            "[log-request-id] POST /api/v1/tasks",
            "[log-request-id] 201 (250.0ms)",
        ]

    @pytest.mark.asyncio
    async def test_dispatch_propagates_call_next_exception(self, monkeypatch):
        middleware = LoggingMiddleware(app=AsyncMock())
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/boom",
                "headers": [],
                "query_string": b"",
                "client": ("127.0.0.1", 1234),
                "server": ("testserver", 80),
                "scheme": "http",
                "http_version": "1.1",
            }
        )
        call_next = AsyncMock(side_effect=RuntimeError("boom"))
        info_messages = []

        monkeypatch.setattr("src.middleware.logging_middleware.uuid.uuid4", lambda: "boom-request-id")
        monkeypatch.setattr(
            "src.middleware.logging_middleware.logger.info",
            lambda message: info_messages.append(message),
        )

        with pytest.raises(RuntimeError, match="boom"):
            await middleware.dispatch(request, call_next)

        assert info_messages == ["[boom-request-id] GET /boom"]

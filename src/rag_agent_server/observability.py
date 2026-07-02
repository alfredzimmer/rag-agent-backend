from __future__ import annotations

import json
import logging
import os
import socket
from datetime import datetime, timezone
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_configured = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        span_context = trace.get_current_span().get_span_context()
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": os.getenv("OTEL_SERVICE_NAME", "rag-agent-backend"),
        }
        if span_context.is_valid:
            payload["trace_id"] = format(span_context.trace_id, "032x")
            payload["span_id"] = format(span_context.span_id, "016x")
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key in ("job_id", "conversation_id", "user_id", "document_id"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = str(value)
        return json.dumps(payload, ensure_ascii=True)


def _instrument_clients() -> None:
    HTTPXClientInstrumentor().instrument()
    RedisInstrumentor().instrument()
    PsycopgInstrumentor().instrument()


def configure_telemetry(service_name: str) -> None:
    global _configured
    if _configured:
        return
    _configured = True

    os.environ["OTEL_SERVICE_NAME"] = service_name
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JsonFormatter())
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(log_level)

    if os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true":
        return

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": os.getenv("RAG_AGENT_VERSION", "0.1.0"),
            "service.instance.id": os.getenv("HOSTNAME", socket.gethostname()),
            "deployment.environment.name": os.getenv("DEPLOYMENT_ENVIRONMENT", "development"),
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    set_logger_provider(logger_provider)
    root_logger.addHandler(LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider))

    _instrument_clients()


def instrument_fastapi(app: Any) -> None:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="^/health$")

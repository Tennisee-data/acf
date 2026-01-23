"""Observability agent for production-ready logging, metrics, and tracing.

Injects observability scaffolding into generated projects:
- JSON structured logging with correlation IDs
- Prometheus metrics or OpenTelemetry hooks
- Request tracing middleware
- Grafana dashboard templates
"""

import json
import time
from pathlib import Path

from llm_backend import LLMBackend
from schemas.observability_config import (
    DashboardPanel,
    GeneratedFile,
    GrafanaDashboard,
    HealthEndpoint,
    LoggingConfig,
    MetricsBackend,
    MetricsConfig,
    ObservabilityReport,
    TracingConfig,
)

from .base import AgentInput, AgentOutput, BaseAgent

# Templates for different frameworks
LOGGING_CONFIG_TEMPLATE = '''"""Structured logging configuration.

Provides JSON-formatted logs with correlation IDs for production observability.
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Context variables for request tracking
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured fields."""

    SENSITIVE_FIELDS = {fields}

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }}

        # Add correlation and request IDs
        if correlation_id := correlation_id_var.get():
            log_data["correlation_id"] = correlation_id
        if request_id := request_id_var.get():
            log_data["request_id"] = request_id

        # Add extra fields
        if hasattr(record, "extra_fields"):
            extra = record.extra_fields
            # Mask sensitive fields
            for key, value in extra.items():
                if any(s in key.lower() for s in self.SENSITIVE_FIELDS):
                    log_data[key] = "***MASKED***"
                else:
                    log_data[key] = value

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add source location for errors
        if record.levelno >= logging.ERROR:
            log_data["source"] = {{
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }}

        return json.dumps(log_data, default=str)


class StructuredLogger(logging.Logger):
    """Logger that supports structured extra fields."""

    def _log_with_extra(
        self,
        level: int,
        msg: str,
        args: tuple,
        exc_info: Any = None,
        extra: dict | None = None,
        **kwargs: Any,
    ) -> None:
        if extra is None:
            extra = {{}}
        extra["extra_fields"] = kwargs
        super()._log(level, msg, args, exc_info=exc_info, extra=extra)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_with_extra(logging.INFO, msg, args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_with_extra(logging.WARNING, msg, args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_with_extra(logging.ERROR, msg, args, exc_info=True, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_with_extra(logging.DEBUG, msg, args, **kwargs)


def setup_logging(level: str = "{log_level}") -> None:
    """Configure structured JSON logging."""
    logging.setLoggerClass(StructuredLogger)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers = [handler]

    # Reduce noise from common libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return logging.getLogger(name)  # type: ignore


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def generate_request_id() -> str:
    """Generate a new request ID."""
    return str(uuid.uuid4())[:8]
'''

FASTAPI_MIDDLEWARE_TEMPLATE = '''"""Request tracing and logging middleware for FastAPI."""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .logging_config import (
    correlation_id_var,
    generate_correlation_id,
    generate_request_id,
    get_logger,
    request_id_var,
)

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging and correlation ID propagation."""

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Extract or generate correlation ID
        correlation_id = request.headers.get(
            "X-Correlation-ID", generate_correlation_id()
        )
        request_id = generate_request_id()

        # Set context variables
        correlation_id_var.set(correlation_id)
        request_id_var.set(request_id)

        # Log request start
        start_time = time.perf_counter()
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None,
        )

        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log request completion
            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            # Add correlation headers to response
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise
'''

FLASK_MIDDLEWARE_TEMPLATE = '''"""Request tracing and logging middleware for Flask."""

import time
import uuid
from functools import wraps

from flask import Flask, g, request

from .logging_config import (
    correlation_id_var,
    generate_correlation_id,
    generate_request_id,
    get_logger,
    request_id_var,
)

logger = get_logger(__name__)


def setup_request_logging(app: Flask) -> None:
    """Set up request logging hooks for Flask."""

    @app.before_request
    def before_request():
        # Extract or generate correlation ID
        g.correlation_id = request.headers.get(
            "X-Correlation-ID", generate_correlation_id()
        )
        g.request_id = generate_request_id()
        g.start_time = time.perf_counter()

        # Set context variables
        correlation_id_var.set(g.correlation_id)
        request_id_var.set(g.request_id)

        logger.info(
            "Request started",
            method=request.method,
            path=request.path,
            client_ip=request.remote_addr,
        )

    @app.after_request
    def after_request(response):
        duration_ms = (time.perf_counter() - g.start_time) * 1000

        logger.info(
            "Request completed",
            method=request.method,
            path=request.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        # Add correlation headers
        response.headers["X-Correlation-ID"] = g.correlation_id
        response.headers["X-Request-ID"] = g.request_id

        return response

    @app.errorhandler(Exception)
    def handle_exception(e):
        duration_ms = (time.perf_counter() - g.start_time) * 1000
        logger.error(
            "Request failed",
            method=request.method,
            path=request.path,
            error=str(e),
            duration_ms=round(duration_ms, 2),
        )
        raise
'''

PROMETHEUS_METRICS_TEMPLATE = '''"""Prometheus metrics for application monitoring."""

from prometheus_client import Counter, Histogram, Info, generate_latest
{framework_import}

# Application info
APP_INFO = Info("app", "Application information")
APP_INFO.info({{"version": "1.0.0", "framework": "{framework}"}})

# Request metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets={buckets},
)

# Error metrics
ERROR_COUNT = Counter(
    "http_errors_total",
    "Total HTTP errors",
    ["method", "endpoint", "error_type"],
)

# Business metrics (customize as needed)
OPERATION_COUNT = Counter(
    "business_operations_total",
    "Total business operations",
    ["operation", "status"],
)


{metrics_endpoint}


def record_request(method: str, endpoint: str, status: int, duration: float) -> None:
    """Record a request metric."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


def record_error(method: str, endpoint: str, error_type: str) -> None:
    """Record an error metric."""
    ERROR_COUNT.labels(method=method, endpoint=endpoint, error_type=error_type).inc()


def record_operation(operation: str, status: str = "success") -> None:
    """Record a business operation metric."""
    OPERATION_COUNT.labels(operation=operation, status=status).inc()
'''

FASTAPI_METRICS_ENDPOINT = '''
from fastapi import FastAPI, Response


def setup_metrics(app: FastAPI) -> None:
    """Set up Prometheus metrics endpoint."""

    @app.get("/metrics")
    async def metrics():
        return Response(
            content=generate_latest(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
'''

FLASK_METRICS_ENDPOINT = '''
from flask import Flask, Response


def setup_metrics(app: Flask) -> None:
    """Set up Prometheus metrics endpoint."""

    @app.route("/metrics")
    def metrics():
        return Response(
            generate_latest(),
            mimetype="text/plain; version=0.0.4; charset=utf-8",
        )
'''

GRAFANA_DASHBOARD_TEMPLATE = {
    "annotations": {"list": []},
    "editable": True,
    "fiscalYearStartMonth": 0,
    "graphTooltip": 0,
    "links": [],
    "liveNow": False,
    "panels": [],
    "refresh": "5s",
    "schemaVersion": 38,
    "style": "dark",
    "tags": ["auto-generated"],
    "templating": {"list": []},
    "time": {"from": "now-1h", "to": "now"},
    "timepicker": {},
    "timezone": "",
    "title": "",
    "uid": "",
    "version": 1,
    "weekStart": "",
}


class ObservabilityAgent(BaseAgent):
    """Agent for injecting observability scaffolding.

    Capabilities:
    - Detect web framework (FastAPI, Flask, Django)
    - Generate JSON structured logging config
    - Add request tracing middleware
    - Set up Prometheus/OpenTelemetry metrics
    - Generate Grafana dashboard templates
    """

    def __init__(self, llm: LLMBackend) -> None:
        """Initialize the agent."""
        super().__init__(llm)

    def default_system_prompt(self) -> str:
        """Return the default system prompt for observability."""
        return """You are an observability expert. Your role is to:
1. Add structured JSON logging with correlation IDs
2. Integrate metrics (Prometheus/OpenTelemetry)
3. Add request tracing middleware
4. Generate Grafana dashboard templates
5. Ensure production-ready monitoring setup"""

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Generate observability scaffolding.

        Args:
            input_data: Agent input with repo_path

        Returns:
            Agent output with ObservabilityReport
        """
        start_time = time.time()
        repo_path = Path(input_data.repo_path)

        report = ObservabilityReport()

        # Detect framework
        framework = self._detect_framework(repo_path)
        report.framework = framework

        # Configure based on context
        config = input_data.context or {}
        report.logging_config = LoggingConfig(
            level=config.get("log_level", "INFO"),
        )
        report.metrics_config = MetricsConfig(
            enabled=config.get("metrics_enabled", True),
            backend=MetricsBackend(config.get("metrics_backend", "prometheus")),
        )
        report.tracing_config = TracingConfig(
            enabled=config.get("tracing_enabled", True),
            service_name=config.get("service_name", repo_path.name),
        )

        # Generate files based on framework
        if framework == "fastapi":
            self._generate_fastapi_observability(repo_path, report)
        elif framework == "flask":
            self._generate_flask_observability(repo_path, report)
        elif framework == "django":
            self._generate_django_observability(repo_path, report)
        else:
            self._generate_generic_observability(repo_path, report)

        # Generate Grafana dashboard
        self._generate_grafana_dashboard(repo_path, report)

        # Add health endpoints info
        report.health_endpoints = [
            HealthEndpoint(
                path="/health",
                checks=["database", "redis", "external_api"],
                include_details=True,
            ),
            HealthEndpoint(path="/ready", checks=["database"], include_details=False),
            HealthEndpoint(path="/live", checks=[], include_details=False),
        ]

        # Generate summary
        report.summary = self._generate_summary(report)

        duration = time.time() - start_time

        return AgentOutput(
            success=True,
            result=report.model_dump(),
            duration_seconds=duration,
        )

    def _detect_framework(self, repo_path: Path) -> str:
        """Detect the web framework used in the project."""
        # Check requirements.txt
        requirements = repo_path / "requirements.txt"
        pyproject = repo_path / "pyproject.toml"

        content = ""
        if requirements.exists():
            content += requirements.read_text().lower()
        if pyproject.exists():
            content += pyproject.read_text().lower()

        # Check for framework indicators
        if "fastapi" in content:
            return "fastapi"
        elif "flask" in content:
            return "flask"
        elif "django" in content:
            return "django"

        # Check source files
        for py_file in repo_path.rglob("*.py"):
            try:
                file_content = py_file.read_text()
                if "from fastapi" in file_content or "import fastapi" in file_content:
                    return "fastapi"
                elif "from flask" in file_content or "import flask" in file_content:
                    return "flask"
                elif "from django" in file_content or "import django" in file_content:
                    return "django"
            except (OSError, UnicodeDecodeError):
                continue

        return "generic"

    def _find_app_directory(self, repo_path: Path) -> Path:
        """Find the main application directory."""
        # Common patterns
        for pattern in ["app", "src", "api", repo_path.name]:
            candidate = repo_path / pattern
            if candidate.is_dir() and (candidate / "__init__.py").exists():
                return candidate

        # If no app directory, use repo root
        return repo_path

    def _generate_fastapi_observability(
        self, repo_path: Path, report: ObservabilityReport
    ) -> None:
        """Generate observability for FastAPI projects."""
        app_dir = self._find_app_directory(repo_path)

        # Generate logging config
        logging_content = LOGGING_CONFIG_TEMPLATE.format(
            fields=repr(report.logging_config.mask_sensitive_fields),
            log_level=report.logging_config.level.value,
        )
        logging_file = app_dir / "logging_config.py"
        logging_file.write_text(logging_content)
        report.files_generated.append(
            GeneratedFile(
                path=str(logging_file.relative_to(repo_path)),
                content=logging_content,
                description="JSON structured logging with correlation IDs",
            )
        )

        # Generate middleware
        middleware_file = app_dir / "middleware.py"
        middleware_file.write_text(FASTAPI_MIDDLEWARE_TEMPLATE)
        report.files_generated.append(
            GeneratedFile(
                path=str(middleware_file.relative_to(repo_path)),
                content=FASTAPI_MIDDLEWARE_TEMPLATE,
                description="Request logging and correlation ID middleware",
            )
        )

        # Generate metrics
        if report.metrics_config.enabled:
            metrics_content = PROMETHEUS_METRICS_TEMPLATE.format(
                framework="fastapi",
                framework_import="from fastapi import FastAPI, Response",
                buckets=repr(report.metrics_config.histogram_buckets),
                metrics_endpoint=FASTAPI_METRICS_ENDPOINT,
            )
            metrics_file = app_dir / "metrics.py"
            metrics_file.write_text(metrics_content)
            report.files_generated.append(
                GeneratedFile(
                    path=str(metrics_file.relative_to(repo_path)),
                    content=metrics_content,
                    description="Prometheus metrics with /metrics endpoint",
                )
            )
            report.dependencies_added.extend([
                "prometheus-client>=0.17.0",
            ])

        # Add dependencies
        report.dependencies_added.extend([
            "python-json-logger>=2.0.0",
        ])

    def _generate_flask_observability(
        self, repo_path: Path, report: ObservabilityReport
    ) -> None:
        """Generate observability for Flask projects."""
        app_dir = self._find_app_directory(repo_path)

        # Generate logging config
        logging_content = LOGGING_CONFIG_TEMPLATE.format(
            fields=repr(report.logging_config.mask_sensitive_fields),
            log_level=report.logging_config.level.value,
        )
        logging_file = app_dir / "logging_config.py"
        logging_file.write_text(logging_content)
        report.files_generated.append(
            GeneratedFile(
                path=str(logging_file.relative_to(repo_path)),
                content=logging_content,
                description="JSON structured logging with correlation IDs",
            )
        )

        # Generate middleware
        middleware_file = app_dir / "middleware.py"
        middleware_file.write_text(FLASK_MIDDLEWARE_TEMPLATE)
        report.files_generated.append(
            GeneratedFile(
                path=str(middleware_file.relative_to(repo_path)),
                content=FLASK_MIDDLEWARE_TEMPLATE,
                description="Request logging hooks for Flask",
            )
        )

        # Generate metrics
        if report.metrics_config.enabled:
            metrics_content = PROMETHEUS_METRICS_TEMPLATE.format(
                framework="flask",
                framework_import="from flask import Flask, Response",
                buckets=repr(report.metrics_config.histogram_buckets),
                metrics_endpoint=FLASK_METRICS_ENDPOINT,
            )
            metrics_file = app_dir / "metrics.py"
            metrics_file.write_text(metrics_content)
            report.files_generated.append(
                GeneratedFile(
                    path=str(metrics_file.relative_to(repo_path)),
                    content=metrics_content,
                    description="Prometheus metrics with /metrics endpoint",
                )
            )
            report.dependencies_added.append("prometheus-client>=0.17.0")

        report.dependencies_added.append("python-json-logger>=2.0.0")

    def _generate_django_observability(
        self, repo_path: Path, report: ObservabilityReport
    ) -> None:
        """Generate observability for Django projects."""
        app_dir = self._find_app_directory(repo_path)

        # Generate logging config
        logging_content = LOGGING_CONFIG_TEMPLATE.format(
            fields=repr(report.logging_config.mask_sensitive_fields),
            log_level=report.logging_config.level.value,
        )
        logging_file = app_dir / "logging_config.py"
        logging_file.write_text(logging_content)
        report.files_generated.append(
            GeneratedFile(
                path=str(logging_file.relative_to(repo_path)),
                content=logging_content,
                description="JSON structured logging with correlation IDs",
            )
        )

        # Generate Django middleware
        django_middleware = '''"""Request tracing middleware for Django."""

import time
from .logging_config import (
    correlation_id_var,
    generate_correlation_id,
    generate_request_id,
    get_logger,
    request_id_var,
)

logger = get_logger(__name__)


class RequestLoggingMiddleware:
    """Middleware for request logging and correlation ID propagation."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Extract or generate correlation ID
        correlation_id = request.headers.get(
            "X-Correlation-ID", generate_correlation_id()
        )
        request_id = generate_request_id()

        # Set context variables
        correlation_id_var.set(correlation_id)
        request_id_var.set(request_id)

        # Store on request for access in views
        request.correlation_id = correlation_id
        request.request_id = request_id

        start_time = time.perf_counter()

        logger.info(
            "Request started",
            method=request.method,
            path=request.path,
            client_ip=request.META.get("REMOTE_ADDR"),
        )

        response = self.get_response(request)

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Request completed",
            method=request.method,
            path=request.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        # Add correlation headers
        response["X-Correlation-ID"] = correlation_id
        response["X-Request-ID"] = request_id

        return response
'''
        middleware_file = app_dir / "middleware.py"
        middleware_file.write_text(django_middleware)
        report.files_generated.append(
            GeneratedFile(
                path=str(middleware_file.relative_to(repo_path)),
                content=django_middleware,
                description="Django request logging middleware",
            )
        )

        report.dependencies_added.extend([
            "python-json-logger>=2.0.0",
            "django-prometheus>=2.3.0",
        ])

    def _generate_generic_observability(
        self, repo_path: Path, report: ObservabilityReport
    ) -> None:
        """Generate generic observability for unknown frameworks."""
        app_dir = self._find_app_directory(repo_path)

        # Generate logging config
        logging_content = LOGGING_CONFIG_TEMPLATE.format(
            fields=repr(report.logging_config.mask_sensitive_fields),
            log_level=report.logging_config.level.value,
        )
        logging_file = app_dir / "logging_config.py"
        if not app_dir.exists():
            app_dir.mkdir(parents=True, exist_ok=True)
        logging_file.write_text(logging_content)
        report.files_generated.append(
            GeneratedFile(
                path=str(logging_file.relative_to(repo_path)),
                content=logging_content,
                description="JSON structured logging with correlation IDs",
            )
        )

        report.dependencies_added.append("python-json-logger>=2.0.0")

    def _generate_grafana_dashboard(
        self, repo_path: Path, report: ObservabilityReport
    ) -> None:
        """Generate a Grafana dashboard JSON file."""
        service_name = report.tracing_config.service_name or repo_path.name

        # Create dashboard panels
        panels = [
            DashboardPanel(
                title="Request Rate",
                type="graph",
                metric_query='rate(http_requests_total{job="' + service_name + '"}[5m])',
                description="Requests per second",
            ),
            DashboardPanel(
                title="Request Latency (p99)",
                type="graph",
                metric_query=(
                    'histogram_quantile(0.99, rate('
                    f'http_request_duration_seconds_bucket{{job="{service_name}"}}[5m]))'
                ),
                description="99th percentile latency",
            ),
            DashboardPanel(
                title="Error Rate",
                type="graph",
                metric_query='rate(http_errors_total{job="' + service_name + '"}[5m])',
                description="Errors per second",
            ),
            DashboardPanel(
                title="Request Count by Status",
                type="stat",
                metric_query='sum by (status) (http_requests_total{job="' + service_name + '"})',
                description="Total requests by status code",
            ),
            DashboardPanel(
                title="Latency Heatmap",
                type="heatmap",
                metric_query=(
                    f'sum(rate(http_request_duration_seconds_bucket'
                    f'{{job="{service_name}"}}[5m])) by (le)'
                ),
                description="Request latency distribution",
            ),
            DashboardPanel(
                title="Top Endpoints by Latency",
                type="table",
                metric_query=(
                    f'topk(10, histogram_quantile(0.95, sum(rate('
                    f'http_request_duration_seconds_bucket{{job="{service_name}"}}[5m]'
                    f')) by (endpoint, le)))'
                ),
                description="Slowest endpoints",
            ),
        ]

        dashboard = GrafanaDashboard(
            title=f"{service_name} - Application Metrics",
            description=f"Auto-generated dashboard for {service_name}",
            panels=panels,
        )
        report.dashboard = dashboard

        # Generate Grafana JSON
        grafana_json = GRAFANA_DASHBOARD_TEMPLATE.copy()
        grafana_json["title"] = dashboard.title
        grafana_json["uid"] = service_name.replace("-", "_")[:40]

        # Convert panels to Grafana format
        grafana_panels = []
        for i, panel in enumerate(panels):
            grafana_panel = {
                "id": i + 1,
                "title": panel.title,
                "type": "timeseries" if panel.type == "graph" else panel.type,
                "gridPos": {"h": 8, "w": 12, "x": (i % 2) * 12, "y": (i // 2) * 8},
                "targets": [
                    {
                        "expr": panel.metric_query,
                        "refId": "A",
                    }
                ],
                "description": panel.description,
            }
            grafana_panels.append(grafana_panel)

        grafana_json["panels"] = grafana_panels

        # Write dashboard file
        dashboards_dir = repo_path / "dashboards"
        dashboards_dir.mkdir(exist_ok=True)
        dashboard_file = dashboards_dir / f"{service_name}-dashboard.json"
        dashboard_file.write_text(json.dumps(grafana_json, indent=2))

        report.dashboard_file = str(dashboard_file.relative_to(repo_path))
        report.files_generated.append(
            GeneratedFile(
                path=report.dashboard_file,
                content=json.dumps(grafana_json, indent=2),
                description="Grafana dashboard for application metrics",
            )
        )

    def _generate_summary(self, report: ObservabilityReport) -> str:
        """Generate a human-readable summary."""
        parts = [
            f"Framework: {report.framework}",
            f"Files generated: {len(report.files_generated)}",
        ]

        if report.logging_config:
            parts.append(
                f"Logging: {report.logging_config.format.value} format, "
                f"level {report.logging_config.level.value}"
            )

        if report.metrics_config.enabled:
            parts.append(f"Metrics: {report.metrics_config.backend.value}")

        if report.tracing_config.enabled:
            parts.append(f"Tracing: {report.tracing_config.backend.value}")

        if report.dashboard_file:
            parts.append(f"Dashboard: {report.dashboard_file}")

        if report.dependencies_added:
            parts.append(f"Dependencies: {', '.join(report.dependencies_added)}")

        return " | ".join(parts)

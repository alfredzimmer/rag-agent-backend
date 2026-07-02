from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - the locked environment includes PyYAML.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "infra" / "docker-compose.yaml"
DEPLOY_SCRIPT = ROOT / "infra" / "deploy.sh"
PRODUCTION_ENV_EXAMPLE = ROOT / "infra" / "env.production.example"
WORKFLOW = ROOT / ".github" / "workflows" / "deploy.yml"


def load_compose() -> dict:
    if yaml is None:
        raise unittest.SkipTest("PyYAML is not available")
    return yaml.safe_load(COMPOSE_FILE.read_text())


def production_env_values() -> dict[str, str]:
    values = {}
    for line in PRODUCTION_ENV_EXAMPLE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class ProductionSmokeTests(unittest.TestCase):
    def test_production_env_declares_public_origin_and_loopback_contract(self) -> None:
        values = production_env_values()

        self.assertEqual(
            values["CORS_ORIGINS"],
            "https://chat.rag-agent.example,https://pis3.aempro.ca",
        )
        self.assertFalse(any(key.endswith("_TUNNEL_TOKEN") for key in values))
        self.assertEqual(values["RAG_AGENT_HTTP_BIND"], "127.0.0.1")
        self.assertEqual(values["TAILSCALE_FUNNEL_ENABLED"], "true")
        self.assertEqual(values["TAILSCALE_FUNNEL_TARGET"], "9229")
        self.assertEqual(values["INGESTION_MAX_UPLOAD_BYTES"], "52428800")

    def test_deploy_rejects_placeholder_secret_before_runtime_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_bin = Path(directory)
            for command in ("curl", "docker", "flock"):
                path = fake_bin / command
                write_executable(path, "#!/usr/bin/env sh\nexit 99\n")

            env = {
                "RAG_AGENT_ENV_FILE": str(PRODUCTION_ENV_EXAMPLE),
                "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            }
            result = subprocess.run(
                ["bash", str(DEPLOY_SCRIPT)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "Production environment file must set a real JWT_SECRET_KEY value.",
            result.stderr,
        )

    def test_deploy_configures_tailscale_funnel_and_public_health(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            fake_bin = temp_dir / "bin"
            fake_bin.mkdir()
            calls_log = temp_dir / "calls.log"
            env_file = temp_dir / "production.env"
            env_file.write_text(
                PRODUCTION_ENV_EXAMPLE.read_text()
                .replace("replace-with-a-long-random-secret", "jwt-secret-real-value")
                .replace("replace-with-a-url-safe-password", "postgres-secret")
                .replace("replace-with-a-strong-password", "minio-secret")
            )

            fake_command = """#!/usr/bin/env sh
printf '%s %s\n' "$(basename "$0")" "$*" >> "$CALLS_LOG"
exit 0
"""
            for command in ("curl", "docker", "flock"):
                write_executable(fake_bin / command, fake_command)
            write_executable(
                fake_bin / "tailscale",
                """#!/usr/bin/env sh
printf '%s %s\n' "$(basename "$0")" "$*" >> "$CALLS_LOG"
case "$*" in
  status)
    exit 0
    ;;
  "funnel --bg --yes 9229")
    exit 0
    ;;
  "funnel status")
    printf 'Available on the internet:\n'
    printf 'https://rag-agent.tailnet.ts.net\n'
    printf '|-- / proxy http://127.0.0.1:9229\n'
    exit 0
    ;;
  *)
    exit 2
    ;;
esac
""",
            )

            env = {
                "CALLS_LOG": str(calls_log),
                "DEPLOY_ROOT": str(ROOT),
                "STATE_DIR": str(temp_dir / "state"),
                "RAG_AGENT_ENV_FILE": str(env_file),
                "PUBLIC_HEALTH_URL": "https://rag-agent.tailnet.ts.net/health",
                "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            }
            result = subprocess.run(
                ["bash", str(DEPLOY_SCRIPT)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            calls = calls_log.read_text()

        self.assertEqual(
            result.returncode,
            0,
            f"deploy failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("tailscale status", calls)
        self.assertIn("tailscale funnel --bg --yes 9229", calls)
        self.assertIn("tailscale funnel status", calls)
        self.assertIn("http://127.0.0.1:9229/health", calls)
        self.assertIn("https://rag-agent.tailnet.ts.net/health", calls)
        self.assertIn("https://rag-agent.tailnet.ts.net", result.stdout)


class ProductionNetworkUnitTests(unittest.TestCase):
    def test_compose_does_not_include_public_tunnel_service(self) -> None:
        services = load_compose()["services"]

        api = services["api"]
        self.assertIn(
            "${RAG_AGENT_HTTP_BIND:-127.0.0.1}:${RAG_AGENT_HTTP_PORT:-9229}:9229",
            api["ports"],
        )
        for service_name, service in services.items():
            with self.subTest(service=service_name):
                self.assertNotEqual(service.get("profiles"), ["edge"])

    def test_api_and_worker_wait_for_milvus_health(self) -> None:
        services = load_compose()["services"]
        milvus = services["milvus"]

        self.assertEqual(
            milvus["healthcheck"]["test"],
            ["CMD", "curl", "--fail", "http://127.0.0.1:9091/healthz"],
        )
        self.assertEqual(
            services["api"]["depends_on"]["milvus"],
            {"condition": "service_healthy"},
        )
        self.assertEqual(
            services["ingestion-worker"]["depends_on"]["milvus"],
            {"condition": "service_healthy"},
        )

    def test_every_published_port_is_bound_to_loopback(self) -> None:
        services = load_compose()["services"]
        env_values = production_env_values()

        for service_name, service in services.items():
            for port in service.get("ports", []):
                with self.subTest(service=service_name, port=port):
                    self.assertIsInstance(port, str)
                    if port.startswith("${RAG_AGENT_HTTP_BIND:-127.0.0.1}:"):
                        self.assertEqual(env_values["RAG_AGENT_HTTP_BIND"], "127.0.0.1")
                        continue
                    self.assertTrue(
                        port.startswith("127.0.0.1:"),
                        f"{service_name} publishes a non-loopback port: {port}",
                    )

    def test_deploy_script_reconciles_local_stack_and_image_tag(self) -> None:
        script = DEPLOY_SCRIPT.read_text()

        self.assertIn("--profile observability", script)
        self.assertIn("--profile tools", script)
        self.assertIn("tailscale funnel --bg --yes", script)
        self.assertIn("tailscale funnel status", script)
        self.assertIn('export RAG_AGENT_IMAGE_TAG="${RAG_AGENT_IMAGE_TAG:-local}"', script)
        for key in (
            "JWT_SECRET_KEY",
            "POSTGRES_PASSWORD",
            "PG_URI",
            "MINIO_ROOT_PASSWORD",
        ):
            self.assertIn(f"require_env_file_value {key}", script)

    def test_workflow_has_public_url_smoke_and_sha_pinning(self) -> None:
        workflow = WORKFLOW.read_text()

        self.assertIn("RAG_AGENT_IMAGE_TAG: ${{ github.sha }}", workflow)
        self.assertIn(
            "PUBLIC_HEALTH_URL: ${{ vars.PRODUCTION_PUBLIC_HEALTH_URL }}",
            workflow,
        )
        self.assertIn('PUBLIC_HEALTH_URL="$5"', workflow)
        self.assertIn("command -v tailscale >/dev/null", workflow)
        self.assertIn('test -n "$PUBLIC_HEALTH_URL"', workflow)
        self.assertIn("curl --fail --silent --show-error --max-time 10", workflow)


class ProductionNetworkIntegrationTests(unittest.TestCase):
    def test_compose_config_renders_with_production_env(self) -> None:
        if shutil.which("docker") is None:
            self.skipTest("docker CLI is not available")

        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "RAG_AGENT_ENV_FILE": str(PRODUCTION_ENV_EXAMPLE),
        }
        result = subprocess.run(
            [
                "docker",
                "compose",
                "--env-file",
                str(PRODUCTION_ENV_EXAMPLE),
                "--profile",
                "observability",
                "--profile",
                "tools",
                "-f",
                str(COMPOSE_FILE),
                "config",
                "--quiet",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            f"docker compose config failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()

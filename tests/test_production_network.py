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


class ProductionSmokeTests(unittest.TestCase):
    def test_production_env_declares_edge_and_cors_contract(self) -> None:
        values = production_env_values()

        self.assertEqual(
            values["CORS_ORIGINS"],
            "https://chat.edemi.org,https://pis3.aempro.ca",
        )
        self.assertEqual(
            values["CLOUDFLARE_TUNNEL_TOKEN"],
            "replace-with-cloudflare-tunnel-token",
        )
        self.assertEqual(values["EDEMI_HTTP_BIND"], "127.0.0.1")
        self.assertEqual(values["INGESTION_MAX_UPLOAD_BYTES"], "52428800")

    def test_deploy_rejects_placeholder_edge_token_before_runtime_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_bin = Path(directory)
            for command in ("curl", "docker", "flock"):
                path = fake_bin / command
                path.write_text("#!/usr/bin/env sh\nexit 99\n")
                path.chmod(path.stat().st_mode | stat.S_IXUSR)

            env = {
                "EDEMI_ENV_FILE": str(PRODUCTION_ENV_EXAMPLE),
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
            "Production environment file must set a real CLOUDFLARE_TUNNEL_TOKEN value.",
            result.stderr,
        )


class ProductionNetworkUnitTests(unittest.TestCase):
    def test_cloudflared_is_edge_profile_only_and_has_no_published_ports(self) -> None:
        services = load_compose()["services"]
        cloudflared = services["cloudflared"]

        self.assertEqual(cloudflared["image"], "cloudflare/cloudflared:2026.6.1")
        self.assertEqual(cloudflared["profiles"], ["edge"])
        self.assertEqual(cloudflared["command"], ["tunnel", "--no-autoupdate", "run"])
        self.assertNotIn("ports", cloudflared)
        self.assertEqual(
            cloudflared["environment"],
            {"TUNNEL_TOKEN": "${CLOUDFLARE_TUNNEL_TOKEN}"},
        )
        self.assertEqual(
            cloudflared["depends_on"],
            {"api": {"condition": "service_healthy"}},
        )

    def test_every_published_port_is_bound_to_loopback(self) -> None:
        services = load_compose()["services"]
        env_values = production_env_values()

        for service_name, service in services.items():
            for port in service.get("ports", []):
                with self.subTest(service=service_name, port=port):
                    self.assertIsInstance(port, str)
                    if port.startswith("${EDEMI_HTTP_BIND:-127.0.0.1}:"):
                        self.assertEqual(env_values["EDEMI_HTTP_BIND"], "127.0.0.1")
                        continue
                    self.assertTrue(
                        port.startswith("127.0.0.1:"),
                        f"{service_name} publishes a non-loopback port: {port}",
                    )

    def test_deploy_script_reconciles_edge_profile_and_image_tag(self) -> None:
        script = DEPLOY_SCRIPT.read_text()

        self.assertIn("--profile edge", script)
        self.assertIn("cloudflared", script)
        self.assertIn('export EDEMI_IMAGE_TAG="${EDEMI_IMAGE_TAG:-local}"', script)
        for key in (
            "CLOUDFLARE_TUNNEL_TOKEN",
            "JWT_SECRET_KEY",
            "POSTGRES_PASSWORD",
            "PG_URI",
            "MINIO_ROOT_PASSWORD",
        ):
            self.assertIn(f"require_env_file_value {key}", script)

    def test_workflow_has_public_edge_smoke_and_sha_pinning(self) -> None:
        workflow = WORKFLOW.read_text()

        self.assertIn("--profile edge", workflow)
        self.assertIn("EDEMI_IMAGE_TAG: ${{ github.sha }}", workflow)
        self.assertIn("PUBLIC_HEALTH_URL", workflow)
        self.assertIn("https://api.edemi.org/health", workflow)
        self.assertIn("curl --fail --silent --show-error --max-time 10", workflow)


class ProductionNetworkIntegrationTests(unittest.TestCase):
    def test_compose_config_renders_with_edge_profile_and_production_env(self) -> None:
        if shutil.which("docker") is None:
            self.skipTest("docker CLI is not available")

        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "EDEMI_ENV_FILE": str(PRODUCTION_ENV_EXAMPLE),
        }
        result = subprocess.run(
            [
                "docker",
                "compose",
                "--env-file",
                str(PRODUCTION_ENV_EXAMPLE),
                "--profile",
                "edge",
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

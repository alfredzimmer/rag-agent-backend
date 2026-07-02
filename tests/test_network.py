from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

try:
    import yaml
except ImportError:  # pragma: no cover - the dev environment includes PyYAML.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "infra" / "docker-compose.yaml"


def load_compose() -> dict:
    if yaml is None:
        raise unittest.SkipTest("PyYAML is not available")
    return yaml.safe_load(COMPOSE_FILE.read_text())


class ComposeNetworkTests(unittest.TestCase):
    def test_stack_is_reduced_to_the_rag_agent(self) -> None:
        services = load_compose()["services"]
        self.assertEqual(set(services), {"api", "etcd", "minio", "milvus"})

    def test_api_uses_compose_network_addresses(self) -> None:
        api = load_compose()["services"]["api"]

        self.assertEqual(api["environment"]["MILVUS_URI"], "http://milvus:19530")
        self.assertEqual(
            api["environment"]["OLLAMA_HOST"], "http://host.docker.internal:11434"
        )
        self.assertIn("host.docker.internal:host-gateway", api["extra_hosts"])

    def test_api_waits_for_milvus_health(self) -> None:
        services = load_compose()["services"]

        self.assertEqual(
            services["milvus"]["healthcheck"]["test"],
            ["CMD", "curl", "--fail", "http://127.0.0.1:9091/healthz"],
        )
        self.assertEqual(
            services["api"]["depends_on"]["milvus"],
            {"condition": "service_healthy"},
        )

    def test_every_published_port_is_bound_to_loopback(self) -> None:
        services = load_compose()["services"]

        for service_name, service in services.items():
            for port in service.get("ports", []):
                with self.subTest(service=service_name, port=port):
                    self.assertTrue(
                        port.startswith("127.0.0.1:")
                        or port.startswith("${RAG_AGENT_HTTP_BIND:-127.0.0.1}:"),
                        f"{service_name} publishes a non-loopback port: {port}",
                    )


class ServiceTargetTests(unittest.TestCase):
    def test_targets_default_to_localhost(self) -> None:
        from rag_agent_server.main import service_targets

        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                service_targets(),
                {"milvus": ("localhost", 19530), "ollama": ("localhost", 11434)},
            )

    def test_targets_follow_container_environment(self) -> None:
        from rag_agent_server.main import service_targets

        env = {
            "MILVUS_URI": "http://milvus:19530",
            "OLLAMA_HOST": "http://host.docker.internal:11434",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            self.assertEqual(
                service_targets(),
                {
                    "milvus": ("milvus", 19530),
                    "ollama": ("host.docker.internal", 11434),
                },
            )


if __name__ == "__main__":
    unittest.main()

# Skills

## Remote Server Development (DO NOT USE LOCAL DOCKER)

**Never** try to run `docker-compose` locally. All services are already running on the remote server.

### SSH Port Forwarding

Connect to the server with port forwarding via:

```
ssh -N -L 19530:127.0.0.1:19530 -L 5433:127.0.0.1:5433 -L 2379:127.0.0.1:2379 -L 2380:127.0.0.1:2380 -L 9000:127.0.0.1:9000 -L 11434:127.0.0.1:11434 ai-server -f
```

This forwards:
- **19530** -> Milvus vector database
- **5433** -> PostgreSQL
- **2379/2380** -> etcd
- **9000** -> MinIO / S3-compatible storage
- **11434** -> Ollama (LLM inference)

### Current Agent Debugging Mode

Milvus is intentionally bypassed in `src/rag/agent.py` while debugging conversation/follow-up behavior. Do not re-enable Milvus or investigate vector-store connectivity unless explicitly asked; focus on PostgreSQL checkpointing and Ollama response generation.

### If Port Forwarding Fails

If testing the connection after running the SSH command yields a failure (e.g., service unreachable, connection refused), **prompt the user to fix the issue** rather than attempting to start a local Docker instance. Do not fall back to local services.

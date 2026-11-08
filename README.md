# Helios — HR Intelligence Copilot

Helios is an AI-powered HR operations copilot for a synthetic company (Northstar Global). It helps employees plan leave, helps managers review overlapping requests and approvals, and helps HR explore workforce attendance — with role-based access, policy retrieval, and human confirmation before any consequential write.

---

## How to start Helios

### Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Steps

1. **Clone the repository** (or open this project folder).
2. **Create your environment file** in the project root:
   ```bash
    cp .env.example .env
   ```

    On Windows PowerShell:
3. **Edit** `.env` and set at least:
   - `POSTGRES_PASSWORD` — any password you choose (must match the password inside `DATABASE_URL`)
   - `JWT_SECRET` — a random string at least 32 characters

    Optional: add `OPENROUTER_API_KEY` if you want LLM-polished answers. Leave it empty to run with deterministic tool-based answers only.
4. **Start everything** from the project root:
   ```bash
    docker compose up --build
   ```

    First start may take several minutes (images, embedding model download, and database seed).
5. **Open the app**
   - UI: [http://localhost:3000](http://localhost:3000)
   - API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Health: [http://localhost:8000/health](http://localhost:8000/health)
6. **Sign in** with a demo account (password for all accounts: `Helios#2026`)
   | Role | Email |
    | --- | --- |
    | Employee | `aisha.rahman@northstar.example` |
    | Manager | `james.okonkwo@northstar.example` |
    | HR Admin | `helena.vogel@northstar.example` |

### Stop Helios

```bash
docker compose down
```

### Reset the database (fresh seed)

```bash
docker compose down -v
docker compose up --build
```

This deletes the local Postgres volume and recreates the synthetic employees, leave balances, policies, and holidays.

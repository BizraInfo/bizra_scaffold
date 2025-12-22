# BIZRA Unified Ecosystem - Installation Guide

This guide helps you set up the unified BIZRA ecosystem on your local machine using the provided scaffold. It assumes you have access to the scaffold contents (including the `BIZRA_SOT.md` template, the Node0 attestation spec, and the system audit report). Once you migrate your existing modules into this repository and follow the steps below, you should be able to run the system locally and extend it with VS Code, GitHub Copilot, and Claude 4.5.

## 1. Prerequisites

To build and run BIZRA locally you will need the following tools installed:

| Tool | Version | Purpose |
|---|---|---|
| **Git** | >= 2.30 | Clone and manage the monorepo. |
| **Node.js** | 18 LTS or 20 LTS | Runs the API layer and utility scripts. |
| **npm** or **pnpm** | Latest (pnpm recommended) | Manages JavaScript dependencies and workspaces. |
| **Rust** | Stable (1.70+) | Builds the core PoI engine and any Rust modules. |
| **Cargo** | Stable | Rust package manager. |
| **Python 3.10+** | Optional | For tooling scripts or data processing. |
| **Docker** | Latest | To run services like PostgreSQL or Redis in containers. |
| **VS Code** | Latest | Recommended IDE (with Rust, TypeScript, and YAML extensions). |
| **GitHub Copilot** or **Claude 4.x** | Optional | Assist with code completion and reasoning. |

## 2. Repository Setup

1. **Create your GitHub repository.** Fork or clone the scaffold into a new repo on GitHub. For example:
   ```bash
   git clone https://github.com/yourusername/bizra
   cd bizra
   ```
2. **Copy existing code.** Migrate the contents of your various directories (Node0, TaskMaster, Dual-Agentic system, etc.) into appropriate packages in the monorepo structure. Each sub-project can live inside `packages/` or `apps/` depending on its role.
3. **Update `package.json` / workspace configuration.** Adopt a workspace manager like **Nx** or **Turborepo** to manage builds and caching. Define each package and its dependencies in the root configuration.
4. **Fill out the `BIZRA_SOT.md`.** The SOT template in this scaffold must be completed with your actual parameters (token names, PoI weights, invariants, etc.). Treat it as code - commit it to the repo and update it whenever the canonical definitions change.

## 3. Installation and Build

1. **Install JavaScript dependencies.** Run:
   ```bash
   pnpm install   # or npm install
   ```
   This installs dependencies for all workspace packages. If using **Nx**, you can run `pnpm nx graph` to visualize dependency relationships.

2. **Install Rust dependencies.** Within any Rust crate directories (e.g. `packages/poi-engine`), run:
   ```bash
   cargo build --release
   ```
   This builds the PoI engine. If you have Node bindings via NAPI, the build may run automatically via `npm install`.

3. **Run tests.** Ensure your environment is sound by executing:
   ```bash
   pnpm test        # runs JavaScript/TypeScript tests
   cargo test       # runs Rust tests
   ```
   Fix any failing tests before proceeding.

4. **Database and services.** If your unified system requires a database (e.g. PostgreSQL for the hive-mind store) or other services (Redis, etc.), launch them via Docker Compose or Kubernetes. Example:
   ```bash
   docker compose up -d
   ```
   or refer to your Node0 `docker-compose.yml`/`k8s` manifests.

5. **Local development server.** Start the API server (Node0, TaskMaster, or other packages) in development mode. The command will vary depending on your workspace structure, for example:
   ```bash
   pnpm run dev        # Node API
   cargo run           # Rust standalone service
   ```
   Verify that the server is reachable at `http://localhost:3000` (or your configured port). Use `/health` and `/ready` endpoints to confirm readiness.

## 4. Recommended VS Code Setup

1. **Install extensions:** Rust Analyzer, TypeScript/JavaScript language support, Prettier, YAML, Docker, and GitLens. If you plan to use GraphQL or OpenAPI, install appropriate schema plugins.
2. **Enable GitHub Copilot / Claude:** Use Copilot for boilerplate generation and code suggestions. For complex reasoning tasks, integrate **Claude 4.x** via an appropriate extension or API (ensure you respect security and licensing). Always verify AI suggestions against your SOT and tests.
3. **Use integrated terminals:** VS Code's integrated terminal simplifies running `cargo` and `pnpm` commands within the correct workspace.

## 5. CI/CD and Testing

Once your code is in the monorepo:

1. **Set up GitHub Actions or another CI system.** Use the templates in the Node0 repo as a starting point. Ensure you enforce:
   - Linting and type checks
   - Unit and integration tests
   - Formal verification for Rust code (e.g. with Prusti/Creusot)
   - SOT compliance and evidence policy checks
2. **Deploy** to your preferred environment (local cluster, cloud). Use Kubernetes manifests or Docker Compose files from your existing projects. Ensure secrets are managed securely (e.g. via HashiCorp Vault). Expose only necessary ports and endpoints.

## 6. Next Steps

Once this scaffold is installed:

1. **Align your existing modules.** Place each code base into the correct package folder, refactor as needed, and update imports.
2. **Integrate the PoI engine.** Connect your Node API to the Rust PoI engine via FFI or NAPI. Test the end-to-end flow: submit contributions, validate evidence, compute PoI, and update the consensus state.
3. **Merge your AI layers.** Incorporate the Dual-Agentic system (PAT/SAT agents) and integrate local LLM models (DeepSeek, Claude) using the bicameral architecture. Maintain separation between logic (cold core) and presentation (warm surface) to avoid hallucinations.
4. **Continue formalizing.** Expand the SOT with more invariants, update the `Genesis_NodeZero_Attestation_Spec_v1.0.md` as your protocol evolves, and maintain documentation in `docs/`.

Following this guide will set up a robust local environment ready for unified development on BIZRA. Remember to commit early and often, enforce the SOT, and document each decision to maintain Ihsan.

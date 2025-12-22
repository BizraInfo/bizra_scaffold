# BIZRA Unified Scaffold (v0.1)

This folder is a **starter kit** to help unify the scattered BIZRA components into a cohesive monorepo. It contains templates, analysis, and specifications you can use to bootstrap a single source of truth and structure your unified system.

## Contents

| File | Description |
|---|---|
| `report.md` | Evidence-based analysis summarizing the current state of BIZRA repositories, architecture, security, performance, documentation, and recommendations for unification. |
| `SOT_template.md` | A template for your **Single Source of Truth (SOT)** file, where you define canonical names, invariants, PoI parameters, and evidence requirements. Customize this file to reflect your project's agreed rules. |
| `BIZRA_SOT.md` | Draft single source of truth for the unified system, derived from `SOT_template.md`. |
| `EVIDENCE_INDEX.md` | Index of evidence artifacts required to validate claims across documentation and reports. |
| `Genesis_NodeZero_Attestation_Spec_v1.0.md` | Official specification of the Genesis Node Zero attestation protocol (for reference). |
| `README.md` | This file. |

## How to Use

1. **Create a new GitHub repository** (e.g. `bizra-unified`), and clone it to your machine.

2. **Copy the contents of this `bizra_scaffold` folder** into your new repository.

3. **Set up a monorepo structure** using a workspace manager (Nx or Turborepo). For example:

   ```
   bizra-unified/
     apps/
       api/            # REST or GraphQL API entrypoint (Node.js)
     packages/
       poi-core/       # PoI scoring and verification library (Rust/TypeScript)
       blockgraph/     # Consensus and BlockGraph logic
       agent-os/       # Dual-Agentic orchestration (PAT/SAT)
       ...
     docs/             # Documentation site built with Docusaurus (optional)
     BIZRA_SOT.md      # Your customized Single Source of Truth (copy from the template)
     ...
   ```

4. **Fill out `BIZRA_SOT.md`** using `SOT_template.md` as a starting point. Define canonical terms (token names, layer names), invariants (e.g. "every claim must have evidence"), and PoI parameters (weights, thresholds, decay rates). This file becomes the binding contract for your project.

5. **Migrate code from existing directories** (`bizra-genesis-node`, `BIZRA-Dual-Agentic-system-`, `BIZRA-OS`, etc.) into the appropriate packages within your monorepo. Remove duplicate copies and keep only the latest, verified code. Use version control to track history.

6. **Ensure that every claim or metric in your documentation links back to evidence**, such as benchmark logs, test results, or signed attestation files. Your CI pipeline should reject changes that break the SOT or refer to claims without proof.

7. **Run your development environment locally** using VS Code, GitHub Copilot, and Claude 4.5 Opus. Because everything is now unified, you can run tests (`nx test`), build services (`nx build`), and spin up local dev servers (`nx serve api`) from a single entry point.

By following these steps you'll convert years of fragmented development into a cohesive, future-proof system.

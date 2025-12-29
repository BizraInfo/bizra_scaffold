# BIZRA Architecture Diagrams

> **Visual Reference for Elite Practitioners**  
> Comprehensive architecture views of BIZRA's core systems.

---

## Table of Contents

1. [Bicameral Architecture](#1-bicameral-architecture)
2. [SNR Scoring Pipeline](#2-snr-scoring-pipeline)
3. [Ihsān Ethical Constraint Flow](#3-ihsān-ethical-constraint-flow)
4. [PAT-SAT Protocol Flow](#4-pat-sat-protocol-flow)
5. [Giants Protocol Engine](#5-giants-protocol-engine)
6. [Memory System Hierarchy](#6-memory-system-hierarchy)
7. [Graph-of-Thoughts Reasoning](#7-graph-of-thoughts-reasoning)
8. [End-to-End Pipeline](#8-end-to-end-pipeline)

---

## 1. Bicameral Architecture

The Bicameral Architecture separates cold (cryptographic) and warm (AI) computation with a typed membrane boundary.

```mermaid
flowchart TB
    subgraph ColdCore["❄️ ColdCore (Rust)"]
        direction TB
        C1[Ed25519 Signing]
        C2[BLAKE3 Hashing]
        C3[ZK Proof Generation]
        C4[Attestation Engine]
        C1 --> C4
        C2 --> C4
        C3 --> C4
    end

    subgraph Membrane["🔀 Membrane (Typed Boundary)"]
        direction TB
        M1[FFI Binding Layer]
        M2[Schema Validation]
        M3[Type Conversion]
        M4[Error Propagation]
        M1 <--> M2
        M2 <--> M3
        M3 <--> M4
    end

    subgraph WarmSurface["🔥 WarmSurface (Python)"]
        direction TB
        W1[PAT Agent]
        W2[SAT Agent]
        W3[Graph-of-Thoughts]
        W4[Memory System]
        W5[Giants Protocol]
        W1 --> W3
        W2 --> W3
        W3 --> W4
        W3 --> W5
    end

    ColdCore <-->|"FFI Calls"| Membrane
    Membrane <-->|"Typed Messages"| WarmSurface

    style ColdCore fill:#1e3a5f,stroke:#3498db,color:#fff
    style Membrane fill:#4a4a4a,stroke:#f39c12,color:#fff
    style WarmSurface fill:#5a2d0c,stroke:#e74c3c,color:#fff
```

### Invariants

| Layer | Responsibility | Cannot |
|-------|---------------|--------|
| ColdCore | Cryptographic proofs, signing, hashing | Access network, call Python |
| Membrane | Type validation, FFI bridging | Store state, skip validation |
| WarmSurface | AI reasoning, memory, agents | Direct crypto operations |

---

## 2. SNR Scoring Pipeline

Signal-to-Noise Ratio (SNR) scoring ensures high-quality reasoning paths.

```mermaid
flowchart LR
    subgraph Input["📥 Input"]
        I1[Raw Content]
        I2[Context Metadata]
    end

    subgraph Analysis["🔬 Analysis"]
        A1["Signal Extraction<br/>(meaningful content)"]
        A2["Noise Detection<br/>(redundancy, fillers)"]
        A3["Context Weighting"]
    end

    subgraph Scoring["📊 Scoring"]
        S1["SNR Calculation<br/>signal / (signal + noise)"]
        S2["Level Classification"]
        S3["Ihsān Cross-Check"]
    end

    subgraph Output["📤 Output"]
        O1["HIGH ≥ 0.8"]
        O2["MEDIUM ≥ 0.5"]
        O3["LOW < 0.5"]
    end

    I1 --> A1
    I2 --> A3
    A1 --> S1
    A2 --> S1
    A3 --> S1
    S1 --> S2
    S2 --> S3
    S3 --> O1
    S3 --> O2
    S3 --> O3

    style O1 fill:#27ae60,stroke:#1e8449,color:#fff
    style O2 fill:#f39c12,stroke:#d68910,color:#fff
    style O3 fill:#e74c3c,stroke:#c0392b,color:#fff
```

### SNR Thresholds

```python
class SNRLevel(Enum):
    HIGH = "HIGH"      # ≥ 0.8 - Premium quality, prioritized
    MEDIUM = "MEDIUM"  # ≥ 0.5 - Acceptable, standard processing
    LOW = "LOW"        # < 0.5 - Filtered or deprioritized
```

---

## 3. Ihsān Ethical Constraint Flow

Ihsān (إحسان) ensures all operations meet ethical excellence threshold ≥ 0.95.

```mermaid
flowchart TB
    subgraph Request["📨 Incoming Request"]
        R1[Proposal Data]
        R2[Action Type]
        R3[Context]
    end

    subgraph Evaluation["⚖️ Ihsān Evaluation"]
        E1["Harm Assessment<br/>(potential damage)"]
        E2["Beneficence Check<br/>(positive intent)"]
        E3["Fairness Analysis<br/>(equitable treatment)"]
        E4["Transparency Score<br/>(clear reasoning)"]
    end

    subgraph Gate["🚦 Gate Decision"]
        G1{{"Ihsān ≥ 0.95?"}}
        G2["✅ ACCEPT<br/>Proceed to execution"]
        G3["❌ REJECT<br/>REJECT_IHSAN_BELOW_MIN"]
    end

    subgraph Audit["📋 Audit Trail"]
        A1[Score Logged]
        A2[Rejection Reason]
        A3[Evidence Hash]
    end

    R1 --> E1
    R2 --> E2
    R3 --> E3
    R3 --> E4

    E1 --> G1
    E2 --> G1
    E3 --> G1
    E4 --> G1

    G1 -->|"Yes"| G2
    G1 -->|"No"| G3

    G2 --> A1
    G3 --> A2
    A1 --> A3
    A2 --> A3

    style G2 fill:#27ae60,stroke:#1e8449,color:#fff
    style G3 fill:#e74c3c,stroke:#c0392b,color:#fff
    style G1 fill:#3498db,stroke:#2980b9,color:#fff
```

### Fail-Closed Principle

> **Critical**: Ihsān evaluation is **fail-closed**. Unknown inputs default to 0.0 (reject), not 1.0 (accept).

---

## 4. PAT-SAT Protocol Flow

PAT (Prover) constructs proposals; SAT (Verifier) validates and commits.

```mermaid
sequenceDiagram
    participant Client
    participant PAT as PAT Agent<br/>(Prover)
    participant SAT as SAT Agent<br/>(Verifier)
    participant EventLog as Event Log
    participant Memory as Memory System

    Client->>PAT: Request proposal
    
    rect rgb(40, 60, 80)
        Note over PAT: Pre-sign Validation
        PAT->>PAT: Compute Ihsān score
        alt Ihsān < 0.95
            PAT-->>Client: REJECT_IHSAN_BELOW_MIN
        end
    end

    PAT->>PAT: Build PCIEnvelope
    PAT->>PAT: Sign with Ed25519
    PAT->>Memory: Retrieve context (optional)
    PAT->>SAT: Submit envelope

    rect rgb(60, 40, 40)
        Note over SAT: Multi-Gate Verification
        SAT->>SAT: 1. Verify signature
        SAT->>SAT: 2. Check replay (nonce)
        SAT->>SAT: 3. Validate policy hash
        SAT->>SAT: 4. Re-check Ihsān ≥ 0.95
        SAT->>SAT: 5. Verify SNR quality
    end

    alt All gates pass
        SAT->>EventLog: Commit with BLAKE3 hash
        EventLog-->>SAT: CommitReceipt
        SAT-->>Client: SUCCESS + receipt
        SAT->>Memory: Store outcome
    else Any gate fails
        SAT-->>Client: REJECT + code + details
    end
```

### Role Separation

| Agent | CAN | CANNOT |
|-------|-----|--------|
| PAT | Construct envelope, sign, pre-validate | Commit, issue receipt |
| SAT | Verify, commit, issue receipt | Construct proposals |

---

## 5. Giants Protocol Engine

Extract wisdom from reasoning traces, standing on the shoulders of giants.

```mermaid
flowchart TB
    subgraph Input["📚 Knowledge Sources"]
        I1[Reasoning Traces]
        I2[Historical Decisions]
        I3[Cross-Domain Patterns]
    end

    subgraph Extraction["🔍 Wisdom Extraction"]
        E1["Pattern Recognition"]
        E2["Cross-Temporal Bridges"]
        E3["SNR-Weighted Ranking"]
    end

    subgraph HubConcepts["🌐 Hub Concepts"]
        H1["verification"]
        H2["ethics"]
        H3["cryptography"]
        H4["governance"]
        H5["evidence"]
    end

    subgraph Flywheel["🔄 Integrity Flywheel"]
        F1["proofs → ihsan"]
        F2["ihsan → gates"]
        F3["gates → publish"]
        F4["publish → proofs"]
        F1 --> F2 --> F3 --> F4 --> F1
    end

    subgraph Output["💎 Crystallized Wisdom"]
        O1[Wisdom Items]
        O2[Domain Tags]
        O3[Confidence Scores]
    end

    I1 --> E1
    I2 --> E2
    I3 --> E3

    E1 --> H1
    E2 --> H2
    E3 --> H3
    H1 --> Flywheel
    H4 --> Flywheel
    H5 --> Flywheel

    Flywheel --> O1
    O1 --> O2
    O1 --> O3

    style Flywheel fill:#9b59b6,stroke:#8e44ad,color:#fff
```

### Wisdom Types

```python
class WisdomType(Enum):
    PATTERN = "pattern"           # Recurring structure
    PRINCIPLE = "principle"       # Guiding rule
    HEURISTIC = "heuristic"       # Practical shortcut
    ANTI_PATTERN = "anti_pattern" # What to avoid
    BRIDGE = "bridge"             # Cross-domain connection
```

---

## 6. Memory System Hierarchy

Four-tier self-evolving memory with SNR-gated promotion.

```mermaid
flowchart TB
    subgraph Tier1["L1: Working Memory"]
        W1["Short-term buffer"]
        W2["Active context"]
        W3["TTL: seconds-minutes"]
    end

    subgraph Tier2["L2: Episodic Memory"]
        E1["Event sequences"]
        E2["Temporal context"]
        E3["TTL: hours-days"]
    end

    subgraph Tier3["L3: Semantic Memory"]
        S1["Extracted knowledge"]
        S2["Concept relationships"]
        S3["TTL: weeks-months"]
    end

    subgraph Tier4["L4: Procedural Memory"]
        P1["Crystallized patterns"]
        P2["Automated skills"]
        P3["TTL: permanent"]
    end

    Tier1 -->|"SNR ≥ 0.6<br/>consolidate"| Tier2
    Tier2 -->|"SNR ≥ 0.7<br/>abstract"| Tier3
    Tier3 -->|"SNR ≥ 0.8<br/>crystallize"| Tier4

    subgraph Retrieval["🔍 Retrieval"]
        R1[Query]
        R2[Semantic Search]
        R3[Results]
    end

    R1 --> R2
    R2 --> Tier1
    R2 --> Tier2
    R2 --> Tier3
    R2 --> Tier4
    Tier1 --> R3
    Tier2 --> R3
    Tier3 --> R3
    Tier4 --> R3

    style Tier1 fill:#3498db,stroke:#2980b9,color:#fff
    style Tier2 fill:#2ecc71,stroke:#27ae60,color:#fff
    style Tier3 fill:#f39c12,stroke:#d68910,color:#fff
    style Tier4 fill:#9b59b6,stroke:#8e44ad,color:#fff
```

### Memory Operations

| Operation | Description |
|-----------|-------------|
| `remember()` | Store new content with auto-tiering |
| `retrieve()` | Semantic search across all tiers |
| `consolidate()` | Promote memories based on SNR |
| `crystallize()` | Extract patterns to L4 |
| `export_to_json()` | Persist to disk |
| `import_from_json()` | Restore from disk |

---

## 7. Graph-of-Thoughts Reasoning

Adaptive beam search with SNR-weighted pruning.

```mermaid
flowchart TB
    subgraph Query["🎯 Query"]
        Q1[Initial Question]
        Q2[Seed Concepts]
    end

    subgraph Exploration["🌳 Tree Exploration"]
        subgraph Depth0["Depth 0"]
            D0A[Concept A]
            D0B[Concept B]
        end
        subgraph Depth1["Depth 1"]
            D1A[A.1]
            D1B[A.2]
            D1C[B.1]
            D1D[B.2]
        end
        subgraph Depth2["Depth 2"]
            D2A[A.1.a]
            D2B[A.2.a]
            D2C[B.1.a]
        end
    end

    subgraph BeamControl["📊 Adaptive Beam"]
        B1["beam_width = base × clarity"]
        B2["SNR-weighted ranking"]
        B3["Low-SNR pruning"]
    end

    subgraph Bridges["🌉 Cross-Domain Bridges"]
        BR1["Analogy Bridge"]
        BR2["Temporal Bridge"]
        BR3["Causal Bridge"]
    end

    subgraph Output["💡 Reasoning Chains"]
        O1[Chain 1: SNR 0.87]
        O2[Chain 2: SNR 0.82]
        O3[Chain 3: SNR 0.79]
    end

    Q1 --> D0A
    Q2 --> D0B
    D0A --> D1A
    D0A --> D1B
    D0B --> D1C
    D0B --> D1D
    D1A --> D2A
    D1B --> D2B
    D1C --> D2C

    D2A --> BeamControl
    D2B --> BeamControl
    D2C --> BeamControl

    BeamControl --> BR1
    BeamControl --> BR2
    BR1 --> O1
    BR2 --> O2
    BR3 --> O3

    style O1 fill:#27ae60,stroke:#1e8449,color:#fff
    style O2 fill:#2ecc71,stroke:#27ae60,color:#fff
    style O3 fill:#f39c12,stroke:#d68910,color:#fff
```

### Beam Width Adaptation

```python
# Clarity-adaptive beam width
current_beam_width = max(1, int(base_beam_width * clarity))

# Prune to current beam width, not base
thoughts = sorted(thoughts, key=lambda t: t.snr_score, reverse=True)
thoughts = thoughts[:current_beam_width]
```

---

## 8. End-to-End Pipeline

Complete flow from user input to verified output.

```mermaid
flowchart TB
    subgraph UserLayer["👤 User Layer"]
        U1[Chat Input]
        U2[Query]
    end

    subgraph IngestionLayer["📥 Ingestion"]
        I1[Profile Extraction]
        I2[Topic Detection]
        I3[Memory Injection]
    end

    subgraph ReasoningLayer["🧠 Reasoning"]
        R1[Giants Protocol]
        R2[Graph-of-Thoughts]
        R3[SNR Scorer]
        R4[Cross-Domain Bridges]
    end

    subgraph ProposalLayer["📝 Proposal"]
        P1[PAT Agent]
        P2[Memory Context]
        P3[Envelope Construction]
    end

    subgraph VerificationLayer["✅ Verification"]
        V1[SAT Agent]
        V2[5 Gates]
        V3[Event Log]
    end

    subgraph OutputLayer["📤 Output"]
        O1[Evidence Pack]
        O2[Commit Receipt]
        O3[Memory Update]
    end

    U1 --> I1
    U2 --> I2
    I1 --> I3
    I2 --> R1

    R1 --> R2
    R2 --> R3
    R3 --> R4

    R4 --> P1
    I3 --> P2
    P2 --> P1
    P1 --> P3

    P3 --> V1
    V1 --> V2
    V2 -->|"All pass"| V3

    V3 --> O1
    V3 --> O2
    O2 --> O3

    style UserLayer fill:#3498db,stroke:#2980b9,color:#fff
    style ReasoningLayer fill:#9b59b6,stroke:#8e44ad,color:#fff
    style VerificationLayer fill:#27ae60,stroke:#1e8449,color:#fff
```

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ❄️ | Cold computation (cryptographic) |
| 🔥 | Warm computation (AI/ML) |
| 🔀 | Boundary/membrane |
| ✅ | Verification/approval |
| ❌ | Rejection/failure |
| 🔄 | Cyclic/recursive process |
| 📊 | Scoring/metrics |

---

## References

- [PROTOCOL.md](../PROTOCOL.md) - Protocol specification
- [BIZRA_SOT.md](../BIZRA_SOT.md) - Source of Truth
- [SECURITY_MODEL.md](./SECURITY_MODEL.md) - Security architecture
- [VERIFICATION_STANDARD.md](./VERIFICATION_STANDARD.md) - Verification requirements

---

*Generated for BIZRA Genesis Protocol v1.0*

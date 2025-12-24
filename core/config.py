"""
PRODUCTION CONFIGURATION MODULE
================================
Loads environment variables with validation and type safety.
Follows SOT canonical specifications (BIZRA_SOT.md Section 2-4).
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Literal, Optional
import os

class BIZRAConfig(BaseSettings):
    """Elite practitioner-grade configuration with full validation."""
    
    # ═══ SYSTEM ═══
    system_mode: Literal["production", "staging", "development"] = "production"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    workers: int = Field(default=10, ge=1, le=100)
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1024, le=65535)
    
    # ═══ NEO4J DATABASE ═══
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str
    neo4j_database: str = "bizra"
    neo4j_max_connection_pool_size: int = 50
    neo4j_connection_timeout: int = 30
    
    # ═══ REDIS CACHE ═══
    redis_uri: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_max_connections: int = 50
    
    # ═══ PROMETHEUS ═══
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    metrics_path: str = "/metrics"
    
    # ═══ COGNITIVE PARAMETERS ═══
    memory_buffer_size: int = Field(default=7, ge=5, le=9)  # Miller's Law
    l1_capacity: int = 9
    compression_ratio_target: float = Field(default=0.45, ge=0.1, le=0.9)
    l2_decay_rate: float = Field(default=0.95, ge=0.8, le=0.99)
    
    # ═══ FAISS & EMBEDDINGS ═══
    faiss_index_type: Literal["Flat", "IVF", "HNSW"] = "Flat"
    faiss_metric: Literal["L2", "InnerProduct", "Cosine"] = "L2"
    embedding_dim: int = 768
    
    # ═══ HYPERGRAPH ═══
    hypergraph_max_nodes: int = 10000
    rich_club_k: int = 5
    
    # ═══ L5 EXECUTION ═══
    crystallization_threshold: float = 0.95
    tool_timeout_ms: int = 30000
    
    # ═══ QUANTUM SECURITY ═══
    quantum_security_enabled: bool = True
    dilithium_variant: Literal["Dilithium2", "Dilithium3", "Dilithium5"] = "Dilithium5"
    kyber_variant: Literal["Kyber512", "Kyber768", "Kyber1024"] = "Kyber1024"
    key_storage_path: str = "/secure/keys"
    
    # ═══ API SECURITY ═══
    api_secret_key: str = Field(default="", description="API secret for HMAC auth")
    jwt_secret: str = Field(default="", description="JWT signing secret")
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    cors_allowed_origins: str = Field(
        default="",
        description="Comma-separated list of allowed origins (empty = no CORS)"
    )
    require_authentication: bool = Field(
        default=True,
        description="Require JWT auth for protected endpoints"
    )
    
    # ═══ BYZANTINE CONSENSUS ═══
    byzantine_validators: int = Field(default=5, ge=4)
    validation_quorum: int = 3
    
    # ═══ TEMPORAL CHAIN ═══
    temporal_hash_algo: Literal["sha3_256", "sha3_512"] = "sha3_512"
    chain_entropy_threshold: float = 4.0
    
    # ═══ IHSĀN ETHICAL FRAMEWORK (SOT Section 3.1) ═══
    ihsan_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    ihsan_enforcement: Literal["strict", "warning", "disabled"] = "strict"
    
    # Dimension weights (MUST sum to 1.0)
    weight_ikhlas: float = 0.30       # Truthfulness
    weight_karama: float = 0.20       # Dignity
    weight_adl: float = 0.20          # Fairness
    weight_kamal: float = 0.20        # Excellence
    weight_istidama: float = 0.10     # Sustainability
    
    # ═══ PERFORMANCE TARGETS ═══
    target_throughput: int = 238      # Cycles per second
    target_p99_latency_ms: int = 20   # P99 latency target
    target_snr: float = 1.95          # Signal-to-noise ratio
    target_uptime: float = 0.9995     # 99.95% uptime
    
    # ═══ KUBERNETES ═══
    k8s_namespace: str = "bizra-prod"
    k8s_service_name: str = "bizra-cognitive-engine"
    k8s_replicas_min: int = 5
    k8s_replicas_max: int = 20
    k8s_cpu_target: int = 70
    k8s_memory_target: int = 75
    
    # ═══ OBSERVABILITY ═══
    tracing_enabled: bool = True
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    trace_sample_rate: float = 0.1
    
    # ═══ DEVELOPMENT ═══
    debug_mode: bool = False
    auto_reload: bool = False
    enable_docs: bool = False
    enable_profiler: bool = False
    
    @validator("validation_quorum")
    def validate_quorum(cls, v, values):
        """Ensure quorum is 2/3+ of validators (Byzantine requirement)."""
        validators = values.get("byzantine_validators", 5)
        required_quorum = int((validators * 2) / 3) + 1
        if v < required_quorum:
            raise ValueError(f"Quorum must be >= {required_quorum} for {validators} validators")
        return v
    
    @validator("weight_istidama")
    def validate_weight_sum(cls, v, values):
        """Verify Ihsān weights sum to 1.0 (SOT compliance)."""
        total = (
            values.get("weight_ikhlas", 0) +
            values.get("weight_karama", 0) +
            values.get("weight_adl", 0) +
            values.get("weight_kamal", 0) +
            v
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ihsān weights must sum to 1.0, got {total}")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Singleton instance
_config: Optional[BIZRAConfig] = None

def get_config() -> BIZRAConfig:
    """Get or create singleton config instance."""
    global _config
    if _config is None:
        _config = BIZRAConfig()
    return _config

"""
PRODUCTION HTTP API INTERFACE
==============================
Elite Practitioner Grade | FastAPI + Prometheus + JWT Auth

Features:
- RESTful API for cognitive operations
- JWT authentication for protected endpoints (using SecureJWTService)
- Secure CORS configuration
- Prometheus metrics export
- Health checks and readiness probes
- Async operations with proper error handling
- OpenAPI/Swagger documentation
"""

from fastapi import FastAPI, HTTPException, Request, status, Depends, Security
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import asyncio
import time
import secrets
import logging

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Internal imports
from core.config import get_config
from core.security.quantum_security_v2 import QuantumSecurityV2
from core.security.jwt_hardened import (
    SecureJWTService,
    JWTConfig,
    TokenValidationError,
    TokenRevocationError,
)
from core.layers.memory_layers_v2 import (
    L2WorkingMemoryV2,
    L3EpisodicMemoryV2,
    L4SemanticHyperGraphV2
)

logger = logging.getLogger(__name__)

# Initialize config
config = get_config()

# Security scheme
security_scheme = HTTPBearer(auto_error=False)

# Initialize hardened JWT service (singleton)
_jwt_service: Optional[SecureJWTService] = None


def _get_jwt_service() -> SecureJWTService:
    """Get or create the JWT service singleton."""
    global _jwt_service
    if _jwt_service is None:
        if not config.jwt_secret:
            raise ValueError("JWT secret not configured. Set BIZRA_JWT_SECRET environment variable.")
        
        jwt_config = JWTConfig(
            secret=config.jwt_secret,
            algorithm=config.jwt_algorithm,
            issuer="bizra-api",
            audience="bizra-clients",
            access_token_lifetime=timedelta(hours=config.jwt_expiry_hours),
            strict_validation=True,  # Enforce strong secrets
        )
        _jwt_service = SecureJWTService(jwt_config)
        logger.info("Initialized SecureJWTService with hardened settings")
    return _jwt_service


# ============================================================================
# JWT AUTHENTICATION
# ============================================================================

class AuthenticationError(Exception):
    """Authentication failure."""
    pass


def verify_jwt_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token using SecureJWTService.
    
    Uses hardened verification with:
    - Issuer/audience validation
    - JTI uniqueness checks
    - Key rotation support
    - Revocation checking
    """
    try:
        service = _get_jwt_service()
        payload = service.verify_token(token, expected_type="access")
        return payload
    except TokenValidationError as e:
        raise AuthenticationError(str(e))
    except TokenRevocationError:
        raise AuthenticationError("Token has been revoked")
    except Exception as e:
        raise AuthenticationError(f"Token verification failed: {e}")


def create_jwt_token(subject: str, additional_claims: Dict[str, Any] = None) -> str:
    """Create a new JWT token using SecureJWTService."""
    service = _get_jwt_service()
    return service.create_access_token(
        subject=subject,
        claims=additional_claims,
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security_scheme)
) -> Optional[Dict[str, Any]]:
    """
    Dependency to get current authenticated user.
    Returns None if auth is disabled or no token provided.
    """
    if not config.require_authentication:
        return {"sub": "anonymous", "authenticated": False}
    
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        payload = verify_jwt_token(credentials.credentials)
        payload["authenticated"] = True
        return payload
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Request metrics
http_requests_total = Counter(
    'bizra_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'bizra_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

# Cognitive cycle metrics
cognitive_cycles_total = Counter(
    'bizra_cognitive_cycles_total',
    'Total cognitive cycles executed'
)

cognitive_cycle_duration_seconds = Histogram(
    'bizra_cognitive_cycle_duration_seconds',
    'Cognitive cycle duration in seconds'
)

ihsan_score_gauge = Gauge(
    'bizra_ihsan_score',
    'Current Ihsān ethical score'
)

# Layer metrics
l2_compression_ratio = Gauge(
    'bizra_l2_compression_ratio',
    'L2 LZMA compression ratio'
)

l3_episode_count = Gauge(
    'bizra_l3_episode_count',
    'L3 total episodes stored'
)

l4_node_count = Gauge(
    'bizra_l4_node_count',
    'L4 semantic graph node count'
)

# Security metrics
temporal_chain_length = Gauge(
    'bizra_temporal_chain_length',
    'Length of quantum temporal chain'
)

chain_entropy_total = Gauge(
    'bizra_chain_entropy_total',
    'Total entropy in temporal chain'
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CognitiveOperationRequest(BaseModel):
    """Request for cognitive operation."""
    operation_type: str = Field(..., description="Type of cognitive operation")
    content: Dict[str, Any] = Field(..., description="Operation content")
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    ethical_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "operation_type": "decision",
                "content": {"task": "optimize_resource_allocation"},
                "urgency": 0.8,
                "ethical_sensitivity": 0.9
            }
        }


class CognitiveOperationResponse(BaseModel):
    """Response from cognitive operation."""
    status: str
    operation_id: str
    temporal_proof: Dict[str, str]
    metrics: Dict[str, Any]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float
    components: Dict[str, str]


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="BIZRA AEON OMEGA",
    description="Cognitive Sovereignty Platform API",
    version="10.0.0",
    docs_url="/docs" if config.enable_docs else None,
    redoc_url="/redoc" if config.enable_docs else None
)

# CORS middleware - secure configuration
# CRITICAL: Never use allow_origins=["*"] with allow_credentials=True
_cors_origins = [
    origin.strip() 
    for origin in config.cors_allowed_origins.split(",") 
    if origin.strip()
]

if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,  # Explicit origins only
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )
elif config.debug_mode:
    # Development only: allow localhost origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
# In production with no CORS configured, CORS is disabled (no middleware added)

# Application state
class AppState:
    def __init__(self):
        self.start_time = time.time()
        self.security: Optional[QuantumSecurityV2] = None
        self.l2: Optional[L2WorkingMemoryV2] = None
        self.l3: Optional[L3EpisodicMemoryV2] = None
        self.l4: Optional[L4SemanticHyperGraphV2] = None
        self.initialized = False

app.state = AppState()


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup."""
    print("=" * 60)
    print("BIZRA AEON OMEGA v10.0.0 - STARTING")
    print("=" * 60)
    
    # Initialize security
    app.state.security = QuantumSecurityV2(
        key_storage_path=config.key_storage_path
    )
    print("✓ Quantum security initialized")
    
    # Initialize L2
    app.state.l2 = L2WorkingMemoryV2(
        compression_preset=6,
        decay_rate=config.l2_decay_rate,
        target_ratio=config.compression_ratio_target
    )
    print("✓ L2 Working Memory initialized")
    
    # Initialize L3
    app.state.l3 = L3EpisodicMemoryV2(
        embedding_dim=config.embedding_dim,
        index_type=config.faiss_index_type
    )
    print("✓ L3 Episodic Memory initialized")
    
    # Initialize L4
    app.state.l4 = L4SemanticHyperGraphV2(
        neo4j_uri=config.neo4j_uri,
        neo4j_auth=(config.neo4j_user, config.neo4j_password)
    )
    await app.state.l4.initialize()
    print("✓ L4 Semantic HyperGraph initialized")
    
    app.state.initialized = True
    print("✓ All systems operational")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\nShutting down BIZRA AEON OMEGA...")
    
    if app.state.l4:
        await app.state.l4.close()
        print("✓ L4 connection closed")
    
    print("✓ Shutdown complete")


# ============================================================================
# MIDDLEWARE - REQUEST TRACKING
# ============================================================================

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Record metrics
    http_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    http_request_duration_seconds.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "BIZRA AEON OMEGA",
        "version": "10.0.0",
        "status": "operational",
        "docs": "/docs" if config.enable_docs else "disabled"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Kubernetes probes."""
    if not app.state.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not initialized"
        )
    
    uptime = time.time() - app.state.start_time
    
    # Check component status
    components = {
        "security": "healthy" if app.state.security else "unavailable",
        "l2_memory": "healthy" if app.state.l2 else "unavailable",
        "l3_memory": "healthy" if app.state.l3 else "unavailable",
        "l4_graph": "healthy" if app.state.l4 and app.state.l4.driver else "unavailable",
    }
    
    # Verify temporal chain integrity
    if app.state.security:
        chain_ok = app.state.security.verify_chain_integrity()
        components["temporal_chain"] = "healthy" if chain_ok else "corrupted"
    
    all_healthy = all(status == "healthy" for status in components.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="10.0.0",
        uptime_seconds=uptime,
        components=components
    )


@app.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes."""
    if not app.state.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Not ready"
        )
    return {"status": "ready"}


@app.post("/cognitive/operation", response_model=CognitiveOperationResponse)
async def execute_cognitive_operation(
    request: CognitiveOperationRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Execute a cognitive operation with full security and monitoring.
    """
    if not app.state.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not initialized"
        )
    
    start_time = time.time()
    
    try:
        # Secure operation with quantum-temporal proof
        secured = await app.state.security.secure_operation({
            "type": request.operation_type,
            "content": request.content,
            "urgency": request.urgency,
            "ethical_sensitivity": request.ethical_sensitivity
        })
        
        # Record metrics
        cognitive_cycles_total.inc()
        
        duration = time.time() - start_time
        cognitive_cycle_duration_seconds.observe(duration)
        
        # Update gauges
        temporal_chain_length.set(len(app.state.security.temporal_chain))
        chain_entropy_total.set(app.state.security.chain_entropy)
        
        return CognitiveOperationResponse(
            status="success",
            operation_id=secured["temporal_proof"]["temporal_hash"][:16],
            temporal_proof=secured["temporal_proof"],
            metrics={
                "duration_ms": duration * 1000,
                "chain_length": secured["chain_length"],
                "cumulative_entropy": secured["cumulative_entropy"]
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Operation failed: {str(e)}"
        )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint."""
    if not config.prometheus_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics disabled"
        )
    
    # Update gauges before exporting
    if app.state.l2:
        stats = app.state.l2.get_compression_stats()
        if stats["avg_ratio"] > 0:
            l2_compression_ratio.set(stats["avg_ratio"])
    
    if app.state.l3:
        stats = app.state.l3.get_stats()
        l3_episode_count.set(stats["total_episodes"])
    
    if app.state.l4:
        try:
            stats = await app.state.l4.get_stats()
            l4_node_count.set(stats["topology"]["node_count"])
        except (KeyError, AttributeError, RuntimeError) as e:
            logging.debug(f"L4 stats unavailable: {e}")  # Don't fail metrics endpoint
    
    return PlainTextResponse(
        generate_latest().decode('utf-8'),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/stats", response_model=Dict[str, Any])
async def system_stats(user: Dict[str, Any] = Depends(get_current_user)):
    """Get comprehensive system statistics. Requires authentication."""
    if not app.state.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not initialized"
        )
    
    stats = {
        "uptime_seconds": time.time() - app.state.start_time,
        "version": "10.0.0",
    }
    
    # Security stats
    if app.state.security:
        stats["security"] = app.state.security.get_chain_stats()
    
    # L2 stats
    if app.state.l2:
        stats["l2"] = app.state.l2.get_compression_stats()
    
    # L3 stats
    if app.state.l3:
        stats["l3"] = app.state.l3.get_stats()
    
    # L4 stats
    if app.state.l4:
        try:
            stats["l4"] = await app.state.l4.get_stats()
        except Exception as e:
            stats["l4"] = {"error": str(e)}
    
    return stats


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

class TokenRequest(BaseModel):
    """Request for authentication token."""
    api_key: str = Field(..., description="API secret key")
    subject: str = Field(default="api_client", description="Token subject identifier")


class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int


@app.post("/auth/token", response_model=TokenResponse)
async def get_auth_token(request: TokenRequest):
    """
    Exchange API key for JWT token.
    
    This endpoint validates the API secret key and returns a JWT token
    for accessing protected endpoints.
    """
    if not config.api_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication not configured"
        )
    
    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(request.api_key, config.api_secret_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    try:
        token = create_jwt_token(subject=request.subject)
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in_seconds=config.jwt_expiry_hours * 3600
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if config.debug_mode else "An error occurred",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Use full module path to ensure uvicorn can locate the app
    # when running from repository root
    uvicorn.run(
        "core.engine.api:app",
        host=config.host,
        port=config.port,
        workers=config.workers if not config.debug_mode else 1,
        reload=config.auto_reload,
        log_level=config.log_level.lower()
    )

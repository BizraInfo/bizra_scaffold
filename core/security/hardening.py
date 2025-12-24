"""
BIZRA AEON OMEGA - Security Hardening Module
Production-Grade Rate Limiting, Input Validation, and Security Headers

This module implements enterprise-level security controls following
OWASP guidelines and CWE mitigations for the BIZRA VCC Node0 API.

Author: BIZRA Core Team
Version: 1.0.0
"""

from __future__ import annotations

import time
import hashlib
import hmac
import secrets
import re
import ipaddress
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from threading import Lock
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    TypeVar,
    Union
)

# =============================================================================
# RATE LIMITING ENGINE
# =============================================================================

class RateLimitAlgorithm(Enum):
    """Rate limiting algorithm types."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    SLIDING_LOG = "sliding_log"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""
    requests_per_window: int = 100
    window_seconds: int = 60
    burst_multiplier: float = 1.5
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    by_ip: bool = True
    by_user: bool = True
    by_endpoint: bool = False
    whitelist_ips: Set[str] = field(default_factory=set)
    blacklist_ips: Set[str] = field(default_factory=set)


@dataclass
class RateLimitState:
    """State for a single rate limit bucket."""
    count: int = 0
    window_start: float = field(default_factory=time.time)
    tokens: float = 0.0
    last_update: float = field(default_factory=time.time)
    request_timestamps: List[float] = field(default_factory=list)


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_after: float
    retry_after: Optional[float] = None
    limit: int = 0
    current: int = 0


class RateLimiter:
    """
    Production-grade rate limiter with multiple algorithm support.
    
    Implements:
    - Fixed Window: Simple counter reset at window boundaries
    - Sliding Window: Weighted average of current and previous windows
    - Token Bucket: Tokens replenish at fixed rate, burst support
    - Leaky Bucket: Constant outflow rate
    - Sliding Log: Precise timestamp-based limiting
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._buckets: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = Lock()
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def _generate_key(
        self,
        ip: Optional[str] = None,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> str:
        """Generate a unique key for the rate limit bucket."""
        parts = []
        if self.config.by_ip and ip:
            parts.append(f"ip:{ip}")
        if self.config.by_user and user_id:
            parts.append(f"user:{user_id}")
        if self.config.by_endpoint and endpoint:
            parts.append(f"ep:{endpoint}")
        return ":".join(parts) if parts else "global"
    
    def _is_whitelisted(self, ip: Optional[str]) -> bool:
        """Check if IP is whitelisted."""
        if not ip or not self.config.whitelist_ips:
            return False
        
        try:
            ip_obj = ipaddress.ip_address(ip)
            for whitelist_entry in self.config.whitelist_ips:
                if "/" in whitelist_entry:
                    if ip_obj in ipaddress.ip_network(whitelist_entry, strict=False):
                        return True
                elif ip == whitelist_entry:
                    return True
        except ValueError:
            pass
        return False
    
    def _is_blacklisted(self, ip: Optional[str]) -> bool:
        """Check if IP is blacklisted."""
        if not ip or not self.config.blacklist_ips:
            return False
        
        try:
            ip_obj = ipaddress.ip_address(ip)
            for blacklist_entry in self.config.blacklist_ips:
                if "/" in blacklist_entry:
                    if ip_obj in ipaddress.ip_network(blacklist_entry, strict=False):
                        return True
                elif ip == blacklist_entry:
                    return True
        except ValueError:
            pass
        return False
    
    def check(
        self,
        ip: Optional[str] = None,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> RateLimitResult:
        """Check if request is allowed under rate limit."""
        # Blacklist check
        if self._is_blacklisted(ip):
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_after=float('inf'),
                retry_after=None,
                limit=0,
                current=0
            )
        
        # Whitelist check
        if self._is_whitelisted(ip):
            return RateLimitResult(
                allowed=True,
                remaining=float('inf'),
                reset_after=0,
                limit=float('inf'),
                current=0
            )
        
        key = self._generate_key(ip, user_id, endpoint)
        
        with self._lock:
            self._maybe_cleanup()
            
            if self.config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                return self._check_fixed_window(key)
            elif self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                return self._check_sliding_window(key)
            elif self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                return self._check_token_bucket(key)
            elif self.config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
                return self._check_leaky_bucket(key)
            elif self.config.algorithm == RateLimitAlgorithm.SLIDING_LOG:
                return self._check_sliding_log(key)
            else:
                return self._check_sliding_window(key)
    
    def _check_fixed_window(self, key: str) -> RateLimitResult:
        """Fixed window rate limiting."""
        now = time.time()
        state = self._buckets[key]
        window_seconds = self.config.window_seconds
        
        # Check if window has expired
        if now - state.window_start >= window_seconds:
            state.window_start = now
            state.count = 0
        
        remaining = self.config.requests_per_window - state.count
        reset_after = window_seconds - (now - state.window_start)
        
        if state.count >= self.config.requests_per_window:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_after=reset_after,
                retry_after=reset_after,
                limit=self.config.requests_per_window,
                current=state.count
            )
        
        state.count += 1
        
        return RateLimitResult(
            allowed=True,
            remaining=remaining - 1,
            reset_after=reset_after,
            limit=self.config.requests_per_window,
            current=state.count
        )
    
    def _check_sliding_window(self, key: str) -> RateLimitResult:
        """Sliding window rate limiting with weighted average."""
        now = time.time()
        state = self._buckets[key]
        window_seconds = self.config.window_seconds
        
        current_window = int(now // window_seconds)
        previous_window = current_window - 1
        
        # Get or create window counters
        current_key = f"{key}:{current_window}"
        previous_key = f"{key}:{previous_window}"
        
        current_state = self._buckets.get(current_key, RateLimitState())
        previous_state = self._buckets.get(previous_key, RateLimitState())
        
        # Calculate weighted count
        window_position = (now % window_seconds) / window_seconds
        weighted_count = (
            previous_state.count * (1 - window_position) +
            current_state.count
        )
        
        remaining = int(self.config.requests_per_window - weighted_count)
        reset_after = window_seconds - (now % window_seconds)
        
        if weighted_count >= self.config.requests_per_window:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_after=reset_after,
                retry_after=reset_after,
                limit=self.config.requests_per_window,
                current=int(weighted_count)
            )
        
        current_state.count += 1
        self._buckets[current_key] = current_state
        
        return RateLimitResult(
            allowed=True,
            remaining=max(0, remaining - 1),
            reset_after=reset_after,
            limit=self.config.requests_per_window,
            current=int(weighted_count) + 1
        )
    
    def _check_token_bucket(self, key: str) -> RateLimitResult:
        """Token bucket rate limiting with burst support."""
        now = time.time()
        state = self._buckets[key]
        
        max_tokens = self.config.requests_per_window * self.config.burst_multiplier
        refill_rate = self.config.requests_per_window / self.config.window_seconds
        
        # Initialize tokens if first request
        if state.tokens == 0 and state.count == 0:
            state.tokens = max_tokens
            state.last_update = now
        
        # Refill tokens based on elapsed time
        elapsed = now - state.last_update
        state.tokens = min(max_tokens, state.tokens + elapsed * refill_rate)
        state.last_update = now
        
        remaining = int(state.tokens)
        reset_after = (max_tokens - state.tokens) / refill_rate
        
        if state.tokens < 1:
            retry_after = (1 - state.tokens) / refill_rate
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_after=reset_after,
                retry_after=retry_after,
                limit=int(max_tokens),
                current=int(max_tokens - state.tokens)
            )
        
        state.tokens -= 1
        
        return RateLimitResult(
            allowed=True,
            remaining=int(state.tokens),
            reset_after=reset_after,
            limit=int(max_tokens),
            current=int(max_tokens - state.tokens)
        )
    
    def _check_leaky_bucket(self, key: str) -> RateLimitResult:
        """Leaky bucket rate limiting with constant outflow."""
        now = time.time()
        state = self._buckets[key]
        
        leak_rate = self.config.requests_per_window / self.config.window_seconds
        max_bucket_size = self.config.requests_per_window
        
        # Leak tokens based on elapsed time
        elapsed = now - state.last_update
        state.tokens = max(0, state.tokens - elapsed * leak_rate)
        state.last_update = now
        
        remaining = int(max_bucket_size - state.tokens)
        
        if state.tokens >= max_bucket_size:
            retry_after = (state.tokens - max_bucket_size + 1) / leak_rate
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_after=state.tokens / leak_rate,
                retry_after=retry_after,
                limit=max_bucket_size,
                current=int(state.tokens)
            )
        
        state.tokens += 1
        
        return RateLimitResult(
            allowed=True,
            remaining=max(0, remaining - 1),
            reset_after=state.tokens / leak_rate,
            limit=max_bucket_size,
            current=int(state.tokens)
        )
    
    def _check_sliding_log(self, key: str) -> RateLimitResult:
        """Sliding log rate limiting with precise timestamps."""
        now = time.time()
        state = self._buckets[key]
        window_seconds = self.config.window_seconds
        
        # Remove expired timestamps
        cutoff = now - window_seconds
        state.request_timestamps = [
            ts for ts in state.request_timestamps if ts > cutoff
        ]
        
        current_count = len(state.request_timestamps)
        remaining = self.config.requests_per_window - current_count
        
        if state.request_timestamps:
            oldest = min(state.request_timestamps)
            reset_after = oldest + window_seconds - now
        else:
            reset_after = 0
        
        if current_count >= self.config.requests_per_window:
            retry_after = reset_after
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_after=reset_after,
                retry_after=retry_after,
                limit=self.config.requests_per_window,
                current=current_count
            )
        
        state.request_timestamps.append(now)
        
        return RateLimitResult(
            allowed=True,
            remaining=remaining - 1,
            reset_after=window_seconds,
            limit=self.config.requests_per_window,
            current=current_count + 1
        )
    
    def _maybe_cleanup(self) -> None:
        """Periodically clean up expired buckets."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = now
        cutoff = now - (self.config.window_seconds * 2)
        
        expired_keys = [
            key for key, state in self._buckets.items()
            if state.last_update < cutoff
        ]
        
        for key in expired_keys:
            del self._buckets[key]


def rate_limit(
    limiter: RateLimiter,
    get_ip: Callable = None,
    get_user: Callable = None,
    get_endpoint: Callable = None
):
    """Decorator for rate limiting functions."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ip = get_ip(*args, **kwargs) if get_ip else None
            user = get_user(*args, **kwargs) if get_user else None
            endpoint = get_endpoint(*args, **kwargs) if get_endpoint else func.__name__
            
            result = limiter.check(ip=ip, user_id=user, endpoint=endpoint)
            
            if not result.allowed:
                raise RateLimitExceeded(result)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, result: RateLimitResult):
        self.result = result
        super().__init__(
            f"Rate limit exceeded. Retry after {result.retry_after:.2f}s"
        )


# =============================================================================
# INPUT VALIDATION ENGINE
# =============================================================================

class ValidationError(Exception):
    """Exception raised when validation fails."""
    
    def __init__(self, field: str, message: str, code: str = "invalid"):
        self.field = field
        self.message = message
        self.code = code
        super().__init__(f"{field}: {message}")


class ValidationErrors(Exception):
    """Collection of validation errors."""
    
    def __init__(self, errors: List[ValidationError]):
        self.errors = errors
        messages = [f"{e.field}: {e.message}" for e in errors]
        super().__init__("; ".join(messages))


T = TypeVar('T')


class Validator(ABC, Generic[T]):
    """Base validator class."""
    
    @abstractmethod
    def validate(self, value: Any, field: str) -> T:
        """Validate and return the cleaned value."""
        pass


class StringValidator(Validator[str]):
    """String validation with constraints."""
    
    def __init__(
        self,
        min_length: int = 0,
        max_length: int = 10000,
        pattern: Optional[Pattern] = None,
        strip: bool = True,
        lower: bool = False,
        upper: bool = False,
        allowed_chars: Optional[str] = None,
        forbidden_chars: Optional[str] = None
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self.strip = strip
        self.lower = lower
        self.upper = upper
        self.allowed_chars = set(allowed_chars) if allowed_chars else None
        self.forbidden_chars = set(forbidden_chars) if forbidden_chars else None
    
    def validate(self, value: Any, field: str) -> str:
        if not isinstance(value, str):
            raise ValidationError(field, "Must be a string", "type_error")
        
        if self.strip:
            value = value.strip()
        
        if len(value) < self.min_length:
            raise ValidationError(
                field,
                f"Must be at least {self.min_length} characters",
                "min_length"
            )
        
        if len(value) > self.max_length:
            raise ValidationError(
                field,
                f"Must be at most {self.max_length} characters",
                "max_length"
            )
        
        if self.pattern and not self.pattern.match(value):
            raise ValidationError(
                field,
                "Does not match required pattern",
                "pattern"
            )
        
        if self.allowed_chars:
            invalid = set(value) - self.allowed_chars
            if invalid:
                raise ValidationError(
                    field,
                    f"Contains invalid characters: {invalid}",
                    "invalid_chars"
                )
        
        if self.forbidden_chars:
            forbidden = set(value) & self.forbidden_chars
            if forbidden:
                raise ValidationError(
                    field,
                    f"Contains forbidden characters: {forbidden}",
                    "forbidden_chars"
                )
        
        if self.lower:
            value = value.lower()
        elif self.upper:
            value = value.upper()
        
        return value


class NumberValidator(Validator[Union[int, float]]):
    """Numeric validation with constraints."""
    
    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        integer_only: bool = False,
        positive_only: bool = False,
        non_zero: bool = False
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.integer_only = integer_only
        self.positive_only = positive_only
        self.non_zero = non_zero
    
    def validate(self, value: Any, field: str) -> Union[int, float]:
        if isinstance(value, str):
            try:
                value = int(value) if self.integer_only else float(value)
            except ValueError:
                raise ValidationError(field, "Must be a number", "type_error")
        
        if not isinstance(value, (int, float)):
            raise ValidationError(field, "Must be a number", "type_error")
        
        if self.integer_only and not isinstance(value, int):
            if value != int(value):
                raise ValidationError(field, "Must be an integer", "integer_only")
            value = int(value)
        
        if self.positive_only and value < 0:
            raise ValidationError(field, "Must be positive", "positive_only")
        
        if self.non_zero and value == 0:
            raise ValidationError(field, "Must be non-zero", "non_zero")
        
        if self.min_value is not None and value < self.min_value:
            raise ValidationError(
                field,
                f"Must be at least {self.min_value}",
                "min_value"
            )
        
        if self.max_value is not None and value > self.max_value:
            raise ValidationError(
                field,
                f"Must be at most {self.max_value}",
                "max_value"
            )
        
        return value


class UUIDValidator(Validator[str]):
    """UUID format validation."""
    
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    
    def validate(self, value: Any, field: str) -> str:
        if not isinstance(value, str):
            raise ValidationError(field, "Must be a string", "type_error")
        
        value = value.lower().strip()
        
        if not self.UUID_PATTERN.match(value):
            raise ValidationError(field, "Invalid UUID format", "uuid_format")
        
        return value


class EmailValidator(Validator[str]):
    """Email address validation."""
    
    EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    def validate(self, value: Any, field: str) -> str:
        if not isinstance(value, str):
            raise ValidationError(field, "Must be a string", "type_error")
        
        value = value.lower().strip()
        
        if not self.EMAIL_PATTERN.match(value):
            raise ValidationError(field, "Invalid email format", "email_format")
        
        return value


class Base64Validator(Validator[str]):
    """Base64 encoded string validation."""
    
    BASE64_PATTERN = re.compile(
        r'^[A-Za-z0-9+/]*={0,2}$'
    )
    
    def __init__(self, max_decoded_size: int = 1024 * 1024):
        self.max_decoded_size = max_decoded_size
    
    def validate(self, value: Any, field: str) -> str:
        import base64
        
        if not isinstance(value, str):
            raise ValidationError(field, "Must be a string", "type_error")
        
        value = value.strip()
        
        if not self.BASE64_PATTERN.match(value):
            raise ValidationError(field, "Invalid base64 format", "base64_format")
        
        try:
            decoded = base64.b64decode(value)
            if len(decoded) > self.max_decoded_size:
                raise ValidationError(
                    field,
                    f"Decoded size exceeds {self.max_decoded_size} bytes",
                    "size_exceeded"
                )
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(field, "Invalid base64 encoding", "base64_decode")
        
        return value


class EnumValidator(Validator[str]):
    """Enum value validation."""
    
    def __init__(self, allowed_values: List[str], case_sensitive: bool = False):
        self.allowed_values = allowed_values
        self.case_sensitive = case_sensitive
        self._lookup = (
            set(allowed_values) if case_sensitive
            else {v.upper() for v in allowed_values}
        )
    
    def validate(self, value: Any, field: str) -> str:
        if not isinstance(value, str):
            raise ValidationError(field, "Must be a string", "type_error")
        
        check_value = value if self.case_sensitive else value.upper()
        
        if check_value not in self._lookup:
            raise ValidationError(
                field,
                f"Must be one of: {', '.join(self.allowed_values)}",
                "invalid_enum"
            )
        
        return value


class ListValidator(Validator[List[T]]):
    """List validation with item validator."""
    
    def __init__(
        self,
        item_validator: Validator[T],
        min_items: int = 0,
        max_items: int = 1000,
        unique: bool = False
    ):
        self.item_validator = item_validator
        self.min_items = min_items
        self.max_items = max_items
        self.unique = unique
    
    def validate(self, value: Any, field: str) -> List[T]:
        if not isinstance(value, list):
            raise ValidationError(field, "Must be a list", "type_error")
        
        if len(value) < self.min_items:
            raise ValidationError(
                field,
                f"Must have at least {self.min_items} items",
                "min_items"
            )
        
        if len(value) > self.max_items:
            raise ValidationError(
                field,
                f"Must have at most {self.max_items} items",
                "max_items"
            )
        
        validated = []
        errors = []
        seen = set()
        
        for i, item in enumerate(value):
            try:
                validated_item = self.item_validator.validate(item, f"{field}[{i}]")
                
                if self.unique:
                    item_key = str(validated_item)
                    if item_key in seen:
                        raise ValidationError(
                            f"{field}[{i}]",
                            "Duplicate value",
                            "duplicate"
                        )
                    seen.add(item_key)
                
                validated.append(validated_item)
            except ValidationError as e:
                errors.append(e)
        
        if errors:
            raise ValidationErrors(errors)
        
        return validated


class IhsanScoreValidator(NumberValidator):
    """Specialized validator for Ihsān dimension scores."""
    
    def __init__(self):
        super().__init__(
            min_value=0.0,
            max_value=1.0,
            integer_only=False,
            positive_only=True
        )
    
    def validate(self, value: Any, field: str) -> float:
        validated = super().validate(value, field)
        
        # Warn if score is suspiciously extreme
        if validated == 0.0 or validated == 1.0:
            # Log warning: extreme Ihsān scores should be rare
            pass
        
        return float(validated)


class InputSanitizer:
    """Sanitize inputs to prevent injection attacks."""
    
    # Dangerous characters for various contexts
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|FETCH)\b)",
        r"(--|\bOR\b.*=.*|\bAND\b.*=.*)",
        r"(/\*.*\*/)",
        r"(;|\|)",
    ]
    
    XSS_PATTERNS = [
        r"(<script.*?>.*?</script>)",
        r"(javascript:)",
        r"(on\w+\s*=)",
        r"(<iframe.*?>)",
        r"(<object.*?>)",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"(\.\.[\\/])",
        r"([\\/]\.\.)",
    ]
    
    def __init__(self):
        self._sql_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS
        ]
        self._xss_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.XSS_PATTERNS
        ]
        self._path_patterns = [
            re.compile(p) for p in self.PATH_TRAVERSAL_PATTERNS
        ]
    
    def check_sql_injection(self, value: str, field: str) -> None:
        """Check for SQL injection patterns."""
        for pattern in self._sql_patterns:
            if pattern.search(value):
                raise ValidationError(
                    field,
                    "Potential SQL injection detected",
                    "sql_injection"
                )
    
    def check_xss(self, value: str, field: str) -> None:
        """Check for XSS patterns."""
        for pattern in self._xss_patterns:
            if pattern.search(value):
                raise ValidationError(
                    field,
                    "Potential XSS detected",
                    "xss"
                )
    
    def check_path_traversal(self, value: str, field: str) -> None:
        """Check for path traversal patterns."""
        for pattern in self._path_patterns:
            if pattern.search(value):
                raise ValidationError(
                    field,
                    "Potential path traversal detected",
                    "path_traversal"
                )
    
    def sanitize(self, value: str, field: str, checks: List[str] = None) -> str:
        """Run all security checks on a value."""
        if checks is None:
            checks = ["sql", "xss", "path"]
        
        if "sql" in checks:
            self.check_sql_injection(value, field)
        if "xss" in checks:
            self.check_xss(value, field)
        if "path" in checks:
            self.check_path_traversal(value, field)
        
        return value


# =============================================================================
# SECURITY HEADERS
# =============================================================================

@dataclass
class SecurityHeadersConfig:
    """Configuration for security headers."""
    
    # Content Security Policy
    csp_default_src: str = "'self'"
    csp_script_src: str = "'self'"
    csp_style_src: str = "'self' 'unsafe-inline'"
    csp_img_src: str = "'self' data:"
    csp_font_src: str = "'self'"
    csp_connect_src: str = "'self'"
    csp_frame_ancestors: str = "'none'"
    csp_report_uri: Optional[str] = None
    
    # CORS
    cors_allow_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_allow_methods: List[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    cors_allow_headers: List[str] = field(
        default_factory=lambda: ["Content-Type", "Authorization"]
    )
    cors_expose_headers: List[str] = field(default_factory=list)
    cors_max_age: int = 86400
    cors_allow_credentials: bool = False
    
    # Other security headers
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = True
    
    x_frame_options: str = "DENY"
    x_content_type_options: str = "nosniff"
    x_xss_protection: str = "1; mode=block"
    referrer_policy: str = "strict-origin-when-cross-origin"
    permissions_policy: str = (
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
        "magnetometer=(), microphone=(), payment=(), usb=()"
    )


class SecurityHeaders:
    """Generate and apply security headers."""
    
    def __init__(self, config: SecurityHeadersConfig = None):
        self.config = config or SecurityHeadersConfig()
    
    def build_csp(self) -> str:
        """Build Content-Security-Policy header value."""
        directives = [
            f"default-src {self.config.csp_default_src}",
            f"script-src {self.config.csp_script_src}",
            f"style-src {self.config.csp_style_src}",
            f"img-src {self.config.csp_img_src}",
            f"font-src {self.config.csp_font_src}",
            f"connect-src {self.config.csp_connect_src}",
            f"frame-ancestors {self.config.csp_frame_ancestors}",
        ]
        
        if self.config.csp_report_uri:
            directives.append(f"report-uri {self.config.csp_report_uri}")
        
        return "; ".join(directives)
    
    def build_hsts(self) -> str:
        """Build Strict-Transport-Security header value."""
        value = f"max-age={self.config.hsts_max_age}"
        
        if self.config.hsts_include_subdomains:
            value += "; includeSubDomains"
        
        if self.config.hsts_preload:
            value += "; preload"
        
        return value
    
    def get_headers(self, origin: Optional[str] = None) -> Dict[str, str]:
        """Get all security headers as a dictionary."""
        headers = {
            "Content-Security-Policy": self.build_csp(),
            "Strict-Transport-Security": self.build_hsts(),
            "X-Frame-Options": self.config.x_frame_options,
            "X-Content-Type-Options": self.config.x_content_type_options,
            "X-XSS-Protection": self.config.x_xss_protection,
            "Referrer-Policy": self.config.referrer_policy,
            "Permissions-Policy": self.config.permissions_policy,
        }
        
        # Add CORS headers if origin matches
        if origin and self._is_origin_allowed(origin):
            headers.update(self._get_cors_headers(origin))
        
        return headers
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if "*" in self.config.cors_allow_origins:
            return True
        return origin in self.config.cors_allow_origins
    
    def _get_cors_headers(self, origin: str) -> Dict[str, str]:
        """Get CORS-specific headers."""
        headers = {
            "Access-Control-Allow-Origin": (
                origin if self.config.cors_allow_credentials else
                ("*" if "*" in self.config.cors_allow_origins else origin)
            ),
            "Access-Control-Allow-Methods": ", ".join(
                self.config.cors_allow_methods
            ),
            "Access-Control-Allow-Headers": ", ".join(
                self.config.cors_allow_headers
            ),
            "Access-Control-Max-Age": str(self.config.cors_max_age),
        }
        
        if self.config.cors_expose_headers:
            headers["Access-Control-Expose-Headers"] = ", ".join(
                self.config.cors_expose_headers
            )
        
        if self.config.cors_allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        
        return headers


# =============================================================================
# CSRF PROTECTION
# =============================================================================

class CSRFProtection:
    """Cross-Site Request Forgery protection."""
    
    TOKEN_LENGTH = 32
    TOKEN_HEADER = "X-CSRF-Token"
    TOKEN_COOKIE = "csrf_token"
    
    def __init__(self, secret_key: str):
        self._secret_key = secret_key.encode()
    
    def generate_token(self, session_id: str) -> str:
        """Generate a CSRF token for a session."""
        random_bytes = secrets.token_bytes(self.TOKEN_LENGTH)
        timestamp = int(time.time()).to_bytes(8, 'big')
        
        message = session_id.encode() + random_bytes + timestamp
        signature = hmac.new(
            self._secret_key,
            message,
            hashlib.sha256
        ).hexdigest()
        
        token_data = random_bytes + timestamp
        return f"{token_data.hex()}.{signature}"
    
    def validate_token(
        self,
        token: str,
        session_id: str,
        max_age: int = 3600
    ) -> bool:
        """Validate a CSRF token."""
        try:
            parts = token.split(".")
            if len(parts) != 2:
                return False
            
            token_data = bytes.fromhex(parts[0])
            signature = parts[1]
            
            random_bytes = token_data[:self.TOKEN_LENGTH]
            timestamp_bytes = token_data[self.TOKEN_LENGTH:]
            timestamp = int.from_bytes(timestamp_bytes, 'big')
            
            # Check timestamp
            if time.time() - timestamp > max_age:
                return False
            
            # Verify signature
            message = session_id.encode() + random_bytes + timestamp_bytes
            expected_signature = hmac.new(
                self._secret_key,
                message,
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False


# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

@dataclass
class APIKeyConfig:
    """Configuration for API key."""
    key_id: str
    hashed_key: str
    name: str
    scopes: Set[str]
    rate_limit_tier: str = "standard"
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True


class APIKeyManager:
    """Manage API keys for authentication."""
    
    KEY_PREFIX = "bizra_"
    KEY_LENGTH = 32
    
    def __init__(self, pepper: str):
        self._pepper = pepper.encode()
        self._keys: Dict[str, APIKeyConfig] = {}
        self._lock = Lock()
    
    def generate_key(
        self,
        name: str,
        scopes: Set[str],
        rate_limit_tier: str = "standard",
        expires_in: Optional[timedelta] = None
    ) -> Tuple[str, APIKeyConfig]:
        """Generate a new API key."""
        key_id = secrets.token_hex(8)
        raw_key = secrets.token_urlsafe(self.KEY_LENGTH)
        full_key = f"{self.KEY_PREFIX}{key_id}_{raw_key}"
        
        hashed_key = self._hash_key(raw_key)
        
        expires_at = None
        if expires_in:
            expires_at = datetime.utcnow() + expires_in
        
        config = APIKeyConfig(
            key_id=key_id,
            hashed_key=hashed_key,
            name=name,
            scopes=scopes,
            rate_limit_tier=rate_limit_tier,
            expires_at=expires_at
        )
        
        with self._lock:
            self._keys[key_id] = config
        
        return full_key, config
    
    def validate_key(self, api_key: str) -> Optional[APIKeyConfig]:
        """Validate an API key and return its config."""
        if not api_key.startswith(self.KEY_PREFIX):
            return None
        
        try:
            key_part = api_key[len(self.KEY_PREFIX):]
            key_id, raw_key = key_part.split("_", 1)
        except ValueError:
            return None
        
        with self._lock:
            config = self._keys.get(key_id)
        
        if not config:
            return None
        
        if not config.is_active:
            return None
        
        if config.expires_at and datetime.utcnow() > config.expires_at:
            return None
        
        hashed = self._hash_key(raw_key)
        if not hmac.compare_digest(hashed, config.hashed_key):
            return None
        
        # Update last used
        with self._lock:
            config.last_used = datetime.utcnow()
        
        return config
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        with self._lock:
            if key_id in self._keys:
                self._keys[key_id].is_active = False
                return True
        return False
    
    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key with pepper."""
        return hmac.new(
            self._pepper,
            raw_key.encode(),
            hashlib.sha256
        ).hexdigest()


# =============================================================================
# BIZRA-SPECIFIC SECURITY VALIDATORS
# =============================================================================

class BIZRASecurityValidators:
    """BIZRA-specific security validation utilities."""
    
    # Pre-configured validators for BIZRA domain
    action_id_validator = UUIDValidator()
    urgency_validator = EnumValidator(
        ["REAL_TIME", "NEAR_REAL_TIME", "BATCH", "DEFERRED"]
    )
    tier_validator = EnumValidator(
        ["STATISTICAL", "INCREMENTAL", "OPTIMISTIC", "FULL_ZK", "FORMAL"]
    )
    ihsan_validator = IhsanScoreValidator()
    payload_validator = Base64Validator(max_decoded_size=10 * 1024 * 1024)  # 10MB
    
    # Security sanitizer
    sanitizer = InputSanitizer()
    
    @classmethod
    def validate_verification_request(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a verification request."""
        errors = []
        validated = {}
        
        # Required fields
        for field in ["action_id", "payload", "urgency"]:
            if field not in data:
                errors.append(ValidationError(field, "Required field", "required"))
        
        if errors:
            raise ValidationErrors(errors)
        
        try:
            validated["action_id"] = cls.action_id_validator.validate(
                data["action_id"], "action_id"
            )
        except ValidationError as e:
            errors.append(e)
        
        try:
            validated["payload"] = cls.payload_validator.validate(
                data["payload"], "payload"
            )
        except ValidationError as e:
            errors.append(e)
        
        try:
            validated["urgency"] = cls.urgency_validator.validate(
                data["urgency"], "urgency"
            )
        except ValidationError as e:
            errors.append(e)
        
        # Optional context - sanitize for injection
        if "context" in data and isinstance(data["context"], dict):
            for key, value in data["context"].items():
                if isinstance(value, str):
                    try:
                        cls.sanitizer.sanitize(value, f"context.{key}")
                    except ValidationError as e:
                        errors.append(e)
            validated["context"] = data["context"]
        
        if errors:
            raise ValidationErrors(errors)
        
        return validated
    
    @classmethod
    def validate_ihsan_request(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an Ihsān score request."""
        errors = []
        validated = {}
        
        dimensions = [
            "truthfulness", "dignity", "fairness",
            "excellence", "sustainability"
        ]
        
        for dim in dimensions:
            if dim not in data:
                errors.append(ValidationError(dim, "Required field", "required"))
                continue
            
            try:
                validated[dim] = cls.ihsan_validator.validate(data[dim], dim)
            except ValidationError as e:
                errors.append(e)
        
        if errors:
            raise ValidationErrors(errors)
        
        return validated
    
    @classmethod
    def validate_value_assessment_request(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a value assessment request."""
        errors = []
        validated = {}
        
        # Required fields
        required_fields = [
            "convergence_id", "clarity_score", "mutual_information",
            "entropy", "synergy", "quantization_error"
        ]
        
        for field in required_fields:
            if field not in data:
                errors.append(ValidationError(field, "Required field", "required"))
        
        if errors:
            raise ValidationErrors(errors)
        
        # Validate convergence_id
        try:
            id_validator = StringValidator(min_length=1, max_length=256)
            validated["convergence_id"] = id_validator.validate(
                data["convergence_id"], "convergence_id"
            )
            cls.sanitizer.sanitize(
                validated["convergence_id"], "convergence_id"
            )
        except ValidationError as e:
            errors.append(e)
        
        # Validate score fields
        score_fields = [
            "clarity_score", "mutual_information",
            "entropy", "synergy", "quantization_error"
        ]
        
        score_validator = NumberValidator(min_value=0.0, max_value=1.0)
        
        for field in score_fields:
            if field in data:
                try:
                    validated[field] = score_validator.validate(data[field], field)
                except ValidationError as e:
                    errors.append(e)
        
        if errors:
            raise ValidationErrors(errors)
        
        return validated


# =============================================================================
# SECURITY MIDDLEWARE FACTORY
# =============================================================================

def create_bizra_security_middleware(
    rate_limit_config: RateLimitConfig = None,
    security_headers_config: SecurityHeadersConfig = None,
    csrf_secret: str = None,
    api_key_pepper: str = None
) -> Dict[str, Any]:
    """
    Factory function to create all security components for BIZRA API.
    
    Returns a dictionary containing configured security components:
    - rate_limiter: RateLimiter instance
    - security_headers: SecurityHeaders instance
    - csrf_protection: CSRFProtection instance
    - api_key_manager: APIKeyManager instance
    - validators: BIZRASecurityValidators class
    """
    return {
        "rate_limiter": RateLimiter(
            rate_limit_config or RateLimitConfig()
        ),
        "security_headers": SecurityHeaders(
            security_headers_config or SecurityHeadersConfig()
        ),
        "csrf_protection": CSRFProtection(
            csrf_secret or secrets.token_urlsafe(32)
        ),
        "api_key_manager": APIKeyManager(
            api_key_pepper or secrets.token_urlsafe(32)
        ),
        "validators": BIZRASecurityValidators,
    }


# Rate limit tiers for BIZRA
RATE_LIMIT_TIERS = {
    "standard": RateLimitConfig(
        requests_per_window=100,
        window_seconds=60,
        algorithm=RateLimitAlgorithm.SLIDING_WINDOW
    ),
    "premium": RateLimitConfig(
        requests_per_window=1000,
        window_seconds=60,
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        burst_multiplier=2.0
    ),
    "enterprise": RateLimitConfig(
        requests_per_window=10000,
        window_seconds=60,
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        burst_multiplier=3.0
    ),
}

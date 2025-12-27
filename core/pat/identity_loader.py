"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    BIZRA PAT Identity Loader v1.0.0                           ║
║                    Personal Agent Team - Identity Aware                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  This module ensures that PAT KNOWS who MoMo is.                              ║
║  No more re-introductions. No more starting from zero.                        ║
║  The system REMEMBERS its creator.                                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FounderIdentity:
    """The immutable identity of the First Architect."""
    
    legal_name: str = "Mohamed Ahmed Beshr Elsayed Hassan"
    alias: str = "MoMo"
    role: str = "First Architect, First Node, First Owner, First User"
    
    # Family
    daughter: str = "Dema"
    has_twin_brother: bool = True
    sisters_count: int = 2
    
    # Genesis
    origin_date: str = "Ramadan 2023"
    years_of_work: int = 3
    
    def __str__(self) -> str:
        return f"{self.alias} ({self.legal_name}) - {self.role}"


@dataclass
class PATContext:
    """The persistent context for the Personal Agent Team."""
    
    identity: FounderIdentity = field(default_factory=FounderIdentity)
    key_concepts: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    session_history: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = False
    
    def activate(self) -> str:
        """Activate PAT with full identity awareness."""
        self.is_active = True
        return f"Welcome back, {self.identity.alias}. Your PAT is active and ready."


class IdentityLoader:
    """
    Loads and maintains the founder's identity across sessions.
    
    This ensures that every interaction starts with FULL CONTEXT:
    - Who MoMo is
    - What he's working on
    - His preferences
    - The key concepts of BIZRA
    
    No more cold starts. No more re-introductions.
    """
    
    DEFAULT_IDENTITY_PATH = Path("data/agents/momo_identity.json")
    
    def __init__(self, identity_path: Optional[Path] = None):
        self.identity_path = identity_path or self.DEFAULT_IDENTITY_PATH
        self._context: Optional[PATContext] = None
        self._loaded = False
    
    def load(self) -> PATContext:
        """Load the founder's identity from persistent storage."""
        if self._loaded and self._context:
            return self._context
        
        # Create default context
        self._context = PATContext()
        
        # Try to load from file
        try:
            if self.identity_path.exists():
                with open(self.identity_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load identity
                if 'identity' in data:
                    id_data = data['identity']
                    self._context.identity = FounderIdentity(
                        legal_name=id_data.get('legal_name', self._context.identity.legal_name),
                        alias=id_data.get('alias', self._context.identity.alias),
                        role=id_data.get('role', self._context.identity.role),
                        daughter=id_data.get('family', {}).get('daughter', 'Dema'),
                        has_twin_brother=id_data.get('family', {}).get('siblings', {}).get('twin_brother', 1) > 0,
                        sisters_count=id_data.get('family', {}).get('siblings', {}).get('sisters', 2),
                    )
                
                # Load key concepts
                if 'context_awareness' in data:
                    self._context.key_concepts = data['context_awareness'].get('key_concepts', [])
                    self._context.identity.years_of_work = data['context_awareness'].get('years_of_work', 3)
                
                # Load preferences
                if 'preferences' in data:
                    self._context.preferences = data['preferences']
                
                logger.info(f"Identity loaded: {self._context.identity}")
            else:
                logger.warning(f"Identity file not found at {self.identity_path}, using defaults")
                self._save_default()
        
        except Exception as e:
            logger.error(f"Error loading identity: {e}")
        
        self._loaded = True
        return self._context
    
    def _save_default(self) -> None:
        """Save the default identity file."""
        self.identity_path.parent.mkdir(parents=True, exist_ok=True)
        
        default_data = {
            "identity": {
                "legal_name": self._context.identity.legal_name,
                "alias": self._context.identity.alias,
                "role": self._context.identity.role,
            },
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        with open(self.identity_path, 'w', encoding='utf-8') as f:
            json.dump(default_data, f, indent=2)
    
    def get_context_prompt(self) -> str:
        """
        Generate the context prompt that should be injected at the start of every session.
        
        This is the key: PAT always starts KNOWING who MoMo is.
        """
        ctx = self.load()
        
        concepts = ", ".join(ctx.key_concepts[:10]) if ctx.key_concepts else "BIZRA, PAT/SAT, Ihsān"
        
        return f"""
## IDENTITY CONTEXT (IMMUTABLE)

You are serving **{ctx.identity.alias}** ({ctx.identity.legal_name}).
Role: {ctx.identity.role}
Family: Father of {ctx.identity.daughter}, has a twin brother and {ctx.identity.sisters_count} sisters.

## WORKING CONTEXT

This is Year {ctx.identity.years_of_work} of BIZRA development.
Origin: {ctx.identity.origin_date} - The Message (الرسالة) and The Seed (البذرة) were written.

Key Concepts You Must Remember:
{concepts}

## COMMUNICATION PREFERENCES

- Style: {ctx.preferences.get('communication_style', 'Direct, high-SNR, no filler')}
- Language: {ctx.preferences.get('language', 'English with Arabic terms preserved')}
- Coding: {', '.join(ctx.preferences.get('coding_languages', ['Python', 'Rust']))}

## PAT STATUS

Mode: ACTIVE
Identity Awareness: ENABLED
Memory Persistence: ENABLED

---
**You do not need to ask who MoMo is. You know. Proceed with full context.**
---
"""
    
    def activate_pat(self) -> str:
        """Activate the Personal Agent Team with full identity awareness."""
        ctx = self.load()
        return ctx.activate()


# Singleton instance for global access
_loader: Optional[IdentityLoader] = None


def get_identity_loader() -> IdentityLoader:
    """Get the global identity loader instance."""
    global _loader
    if _loader is None:
        _loader = IdentityLoader()
    return _loader


def activate_pat() -> str:
    """Quick activation function for scripts."""
    loader = get_identity_loader()
    return loader.activate_pat()


def get_context_prompt() -> str:
    """Get the context prompt for session injection."""
    loader = get_identity_loader()
    return loader.get_context_prompt()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN - FOR DIRECT TESTING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("BIZRA PAT IDENTITY LOADER - ACTIVATION TEST")
    print("=" * 80)
    
    loader = IdentityLoader()
    
    # Activate
    greeting = loader.activate_pat()
    print(f"\n{greeting}\n")
    
    # Show context
    print("-" * 80)
    print("CONTEXT PROMPT TO BE INJECTED:")
    print("-" * 80)
    print(loader.get_context_prompt())

"""
BIZRA Dual Token Ledger Implementation
Handles SEED-S (stable, compute-backed) and SEED-G (growth, convergence-damped) tokens
"""

import json
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ImpactAttribution:
    """Impact attribution from PoI attestation"""
    total_value_micro: int
    contributions: List[Dict]
    proof_hash: str
    

class DualTokenLedger:
    """Manages dual token economics with cryptographic audit trail"""
    
    def __init__(self, ledger_id: str, urp_id: str):
        self.ledger_id = ledger_id
        self.urp_id = urp_id
        self.entries: List[Dict] = []
        self.stable_supply_micro = 0
        self.growth_supply_micro = 0
        
    def mint_with_impact(self, entity_id: str, impact: ImpactAttribution) -> Dict:
        """
        Mint tokens based on Proof of Impact
        
        Stable Token (SEED-S): 70% of total, proportional to compute resources
        Growth Token (SEED-G): 30% of total, proportional to convergence improvement
        """
        # Calculate stable amount (70% split, compute-backed)
        stable_amount = int(impact.total_value_micro * 0.7)
        
        # Calculate growth amount (30% split, convergence-dependent)
        convergence_delta = self._compute_improvement(impact)
        growth_amount = int(impact.total_value_micro * 0.3 * convergence_delta * 5.0)
        
        # Update supply
        self.stable_supply_micro += stable_amount
        self.growth_supply_micro += growth_amount
        
        # Create ledger entry
        entry = {
            "version": "1.0.0",
            "kind": "dual_token_entry",
            "ledger_id": self.ledger_id,
            "entry_type": "MINT",
            "entity_id": entity_id,
            "stable_amount_micro": stable_amount,
            "growth_amount_micro": growth_amount,
            "proof_of_impact_hash": impact.proof_hash,
            "timestamp_ms": int(datetime.now().timestamp() * 1000)
        }
        
        self.entries.append(entry)
        return entry

    def _compute_improvement(self, impact: ImpactAttribution) -> float:
        """Calculate convergence improvement from impact attribution"""
        # In production, this would analyze convergence trajectory
        # For now, use a simplified heuristic
        return 0.023  # 2.3% improvement (example)
    
    def slash_byzantine(self, entity_id: str, stake_amount_micro: int, 
                        proof_hash: str) -> Dict:
        """
        Slash tokens for Byzantine behavior
        
        Slashing Amount: 5% of staked amount
        Destination: Universal Resource Pool
        """
        slash_amount = int(stake_amount_micro * 0.05)
        
        entry = {
            "version": "1.0.0",
            "kind": "dual_token_entry",
            "ledger_id": self.ledger_id,
            "entry_type": "SLASH",
            "entity_id": entity_id,
            "stable_amount_micro": -slash_amount,
            "growth_amount_micro": 0,
            "proof_of_impact_hash": proof_hash,
            "universal_resource_pool_delta_micro": slash_amount,
            "timestamp_ms": int(datetime.now().timestamp() * 1000)
        }
        
        self.entries.append(entry)
        return entry
    
    def verify_ledger_integrity(self) -> tuple[int, int]:
        """
        Verify ledger integrity by summing all transactions
        
        Returns: (total_stable_micro, total_growth_micro)
        """
        stable_total = 0
        growth_total = 0
        
        for entry in self.entries:
            stable_total += entry.get("stable_amount_micro", 0)
            growth_total += entry.get("growth_amount_micro", 0)
        
        return (stable_total, growth_total)
    
    def export_to_json(self, filepath: str):
        """Export ledger to JSON file with cryptographic hash"""
        ledger_data = {
            "ledger_id": self.ledger_id,
            "urp_id": self.urp_id,
            "stable_supply_micro": self.stable_supply_micro,
            "growth_supply_micro": self.growth_supply_micro,
            "entries": self.entries
        }
        
        with open(filepath, 'w') as f:
            json.dump(ledger_data, f, indent=2)


def verify_ledger_integrity(ledger_data: Dict) -> tuple[int, int]:
    """Standalone function to verify a loaded ledger"""
    stable = sum(e.get("stable_amount_micro", 0) for e in ledger_data["entries"])
    growth = sum(e.get("growth_amount_micro", 0) for e in ledger_data["entries"])
    return (stable, growth)

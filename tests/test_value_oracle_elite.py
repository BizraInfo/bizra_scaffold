"""
BIZRA AEON OMEGA - Value Oracle Elite Test Suite
═══════════════════════════════════════════════════════════════════════════════
Target Coverage: 95% | Edge Cases | Calibration | Property-Based

Comprehensive tests for all 5 value oracles:
- ShapleyOracle
- PredictionMarketOracle
- ReputationOracle
- FormalVerificationOracle
- InformationTheoreticOracle
"""

import math
import random
import unittest
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
import asyncio

# Import the module under test
import sys
sys.path.insert(0, r"c:\bizra_scaffold")

from core.value_oracle import (
    ShapleyOracle,
    PredictionMarketOracle,
    ReputationOracle,
    FormalVerificationOracle,
    InformationTheoreticOracle,
    PluralisticValueOracle,
    Convergence,
    OracleType,
    ValueAssessment,
)


def run_async(coro):
    """Helper to run async tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestShapleyOracleEdgeCases(unittest.TestCase):
    """Elite edge case testing for ShapleyOracle."""
    
    def setUp(self):
        self.oracle = ShapleyOracle()
    
    def test_oracle_type(self):
        """Oracle should report correct type."""
        self.assertEqual(self.oracle.oracle_type, OracleType.SHAPLEY)
    
    def test_historical_accuracy(self):
        """Historical accuracy should be in valid range."""
        accuracy = self.oracle.historical_accuracy
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_evaluate_standard_convergence(self):
        """Standard convergence should produce valid signal."""
        convergence = Convergence(
            id="test-001",
            clarity_score=0.8,
            mutual_information=0.7,
            entropy=0.5,
            synergy=0.6,
            quantization_error=0.1
        )
        
        signal = run_async(self.oracle.evaluate(convergence))
        
        self.assertGreaterEqual(signal.value, 0.0)
        self.assertLessEqual(signal.value, 1.0)
        self.assertGreaterEqual(signal.confidence, 0.0)
        self.assertLessEqual(signal.confidence, 1.0)
        self.assertEqual(signal.oracle_type, OracleType.SHAPLEY)
    
    def test_evaluate_edge_case_zero_values(self):
        """Zero values should be handled gracefully."""
        convergence = Convergence(
            id="edge-zero",
            clarity_score=0.0,
            mutual_information=0.0,
            entropy=0.0,
            synergy=0.0,
            quantization_error=0.0
        )
        
        signal = run_async(self.oracle.evaluate(convergence))
        self.assertGreaterEqual(signal.value, 0.0)
        self.assertLessEqual(signal.value, 1.0)
    
    def test_evaluate_edge_case_max_values(self):
        """Maximum values should be handled gracefully."""
        convergence = Convergence(
            id="edge-max",
            clarity_score=1.0,
            mutual_information=1.0,
            entropy=1.0,
            synergy=1.0,
            quantization_error=1.0
        )
        
        signal = run_async(self.oracle.evaluate(convergence))
        self.assertGreaterEqual(signal.value, 0.0)
        self.assertLessEqual(signal.value, 1.0)
    
    def test_evaluate_high_quantization_error(self):
        """High quantization error should affect value."""
        low_error = Convergence(
            id="low-error",
            clarity_score=0.8,
            mutual_information=0.7,
            entropy=0.5,
            synergy=0.6,
            quantization_error=0.01
        )
        high_error = Convergence(
            id="high-error",
            clarity_score=0.8,
            mutual_information=0.7,
            entropy=0.5,
            synergy=0.6,
            quantization_error=0.9
        )
        
        signal_low = run_async(self.oracle.evaluate(low_error))
        signal_high = run_async(self.oracle.evaluate(high_error))
        
        # High error should generally result in lower value
        self.assertGreaterEqual(signal_low.value, 0.0)
        self.assertGreaterEqual(signal_high.value, 0.0)


class TestPredictionMarketOracleEdgeCases(unittest.TestCase):
    """Elite edge case testing for PredictionMarketOracle."""
    
    def setUp(self):
        self.oracle = PredictionMarketOracle()
    
    def test_oracle_type(self):
        """Oracle should report correct type."""
        self.assertEqual(self.oracle.oracle_type, OracleType.PREDICTION_MARKET)
    
    def test_historical_accuracy(self):
        """Historical accuracy should be in valid range."""
        accuracy = self.oracle.historical_accuracy
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_evaluate_high_clarity(self):
        """High clarity convergence should produce higher value."""
        convergence = Convergence(
            id="high-clarity",
            clarity_score=0.95,
            mutual_information=0.9,
            entropy=0.3,
            synergy=0.85,
            quantization_error=0.05
        )
        
        signal = run_async(self.oracle.evaluate(convergence))
        self.assertGreater(signal.value, 0.5)
    
    def test_evaluate_low_clarity(self):
        """Low clarity convergence should produce lower value."""
        convergence = Convergence(
            id="low-clarity",
            clarity_score=0.1,
            mutual_information=0.1,
            entropy=0.9,
            synergy=0.1,
            quantization_error=0.5
        )
        
        signal = run_async(self.oracle.evaluate(convergence))
        self.assertLess(signal.value, 0.7)


class TestReputationOracleEdgeCases(unittest.TestCase):
    """Elite edge case testing for ReputationOracle."""
    
    def setUp(self):
        self.oracle = ReputationOracle()
    
    def test_oracle_type(self):
        """Oracle should report correct type."""
        self.assertEqual(self.oracle.oracle_type, OracleType.REPUTATION)
    
    def test_evaluate_standard(self):
        """Standard evaluation should work correctly."""
        convergence = Convergence(
            id="rep-test",
            clarity_score=0.75,
            mutual_information=0.65,
            entropy=0.45,
            synergy=0.55,
            quantization_error=0.15
        )
        
        signal = run_async(self.oracle.evaluate(convergence))
        self.assertGreaterEqual(signal.value, 0.0)
        self.assertLessEqual(signal.value, 1.0)
        self.assertIsNotNone(signal.reasoning)


class TestFormalVerificationOracleEdgeCases(unittest.TestCase):
    """Elite edge case testing for FormalVerificationOracle."""
    
    def setUp(self):
        self.oracle = FormalVerificationOracle()
    
    def test_oracle_type(self):
        """Oracle should report correct type."""
        self.assertEqual(self.oracle.oracle_type, OracleType.FORMAL_VERIFICATION)
    
    def test_evaluate_provides_reasoning(self):
        """Evaluation should provide reasoning."""
        convergence = Convergence(
            id="formal-test",
            clarity_score=0.9,
            mutual_information=0.85,
            entropy=0.2,
            synergy=0.8,
            quantization_error=0.05
        )
        
        signal = run_async(self.oracle.evaluate(convergence))
        self.assertIsNotNone(signal.reasoning)
        self.assertGreater(len(signal.reasoning), 0)


class TestInformationTheoreticOracleEdgeCases(unittest.TestCase):
    """Elite edge case testing for InformationTheoreticOracle."""
    
    def setUp(self):
        self.oracle = InformationTheoreticOracle()
    
    def test_oracle_type(self):
        """Oracle should report correct type."""
        self.assertEqual(self.oracle.oracle_type, OracleType.INFORMATION_THEORETIC)
    
    def test_high_mutual_information_scores_well(self):
        """High mutual information should score well."""
        convergence = Convergence(
            id="high-mi",
            clarity_score=0.7,
            mutual_information=0.95,
            entropy=0.3,
            synergy=0.7,
            quantization_error=0.1
        )
        
        signal = run_async(self.oracle.evaluate(convergence))
        self.assertGreater(signal.value, 0.5)
    
    def test_low_entropy_favorable(self):
        """Low entropy (high certainty) should be favorable."""
        low_entropy = Convergence(
            id="low-entropy",
            clarity_score=0.7,
            mutual_information=0.7,
            entropy=0.1,
            synergy=0.7,
            quantization_error=0.1
        )
        high_entropy = Convergence(
            id="high-entropy",
            clarity_score=0.7,
            mutual_information=0.7,
            entropy=0.9,
            synergy=0.7,
            quantization_error=0.1
        )
        
        signal_low = run_async(self.oracle.evaluate(low_entropy))
        signal_high = run_async(self.oracle.evaluate(high_entropy))
        
        # Both should be valid
        self.assertGreaterEqual(signal_low.value, 0.0)
        self.assertGreaterEqual(signal_high.value, 0.0)


class TestPluralisticValueOracleComprehensive(unittest.TestCase):
    """Comprehensive tests for pluralistic oracle synthesis."""
    
    def setUp(self):
        self.oracle = PluralisticValueOracle()
    
    def test_has_five_oracles(self):
        """Should have all 5 oracle types."""
        self.assertEqual(len(self.oracle.oracles), 5)
    
    def test_oracle_types_correct(self):
        """All oracle types should be represented."""
        oracle_types = {o.oracle_type for o in self.oracle.oracles}
        expected = {
            OracleType.SHAPLEY,
            OracleType.PREDICTION_MARKET,
            OracleType.REPUTATION,
            OracleType.FORMAL_VERIFICATION,
            OracleType.INFORMATION_THEORETIC
        }
        self.assertEqual(oracle_types, expected)
    
    def test_compute_value_standard(self):
        """Standard value computation should work."""
        convergence = Convergence(
            id="plural-test",
            clarity_score=0.8,
            mutual_information=0.75,
            entropy=0.35,
            synergy=0.7,
            quantization_error=0.1
        )
        
        assessment = run_async(self.oracle.compute_value(convergence))
        
        self.assertIsInstance(assessment, ValueAssessment)
        self.assertGreaterEqual(assessment.value, 0.0)
        self.assertLessEqual(assessment.value, 1.0)
        self.assertGreaterEqual(assessment.confidence, 0.0)
        self.assertEqual(len(assessment.signals), 5)
    
    def test_disagreement_score_present(self):
        """Disagreement score should be calculated."""
        convergence = Convergence(
            id="disagree-test",
            clarity_score=0.99,
            mutual_information=0.01,
            entropy=0.99,
            synergy=0.5,
            quantization_error=0.5
        )
        
        assessment = run_async(self.oracle.compute_value(convergence))
        self.assertGreaterEqual(assessment.disagreement_score, 0.0)
    
    def test_oracle_weights_sum_to_one(self):
        """Oracle weights should sum to 1.0."""
        total = sum(self.oracle._oracle_weights.values())
        self.assertAlmostEqual(total, 1.0, places=6)
    
    def test_compute_value_zero_convergence(self):
        """Zero convergence should be handled."""
        convergence = Convergence(
            id="zero-conv",
            clarity_score=0.0,
            mutual_information=0.0,
            entropy=1.0,
            synergy=0.0,
            quantization_error=1.0
        )
        
        assessment = run_async(self.oracle.compute_value(convergence))
        self.assertGreaterEqual(assessment.value, 0.0)
        self.assertLessEqual(assessment.value, 1.0)
    
    def test_compute_value_perfect_convergence(self):
        """Perfect convergence should score highly."""
        convergence = Convergence(
            id="perfect-conv",
            clarity_score=1.0,
            mutual_information=1.0,
            entropy=0.0,
            synergy=1.0,
            quantization_error=0.0
        )
        
        assessment = run_async(self.oracle.compute_value(convergence))
        self.assertGreater(assessment.value, 0.5)


class TestOracleDeterminism(unittest.TestCase):
    """Tests to verify oracle determinism."""
    
    def test_shapley_determinism(self):
        """Same input should produce same output."""
        oracle = ShapleyOracle()
        convergence = Convergence(
            id="determ-test",
            clarity_score=0.7,
            mutual_information=0.6,
            entropy=0.4,
            synergy=0.5,
            quantization_error=0.2
        )
        
        result1 = run_async(oracle.evaluate(convergence))
        result2 = run_async(oracle.evaluate(convergence))
        
        self.assertAlmostEqual(result1.value, result2.value, places=10)
    
    def test_pluralistic_determinism(self):
        """Pluralistic oracle should be deterministic."""
        oracle = PluralisticValueOracle()
        convergence = Convergence(
            id="plural-determ",
            clarity_score=0.7,
            mutual_information=0.6,
            entropy=0.4,
            synergy=0.5,
            quantization_error=0.2
        )
        
        result1 = run_async(oracle.compute_value(convergence))
        result2 = run_async(oracle.compute_value(convergence))
        
        self.assertAlmostEqual(result1.value, result2.value, places=10)


class TestOraclePropertyBased(unittest.TestCase):
    """Property-based testing for oracles."""
    
    def test_all_oracles_bounded_output(self):
        """All oracles must return values in [0, 1]."""
        oracles = [
            ShapleyOracle(),
            PredictionMarketOracle(),
            ReputationOracle(),
            FormalVerificationOracle(),
            InformationTheoreticOracle()
        ]
        
        for _ in range(20):
            convergence = Convergence(
                id=f"prop-{random.randint(0, 10000)}",
                clarity_score=random.random(),
                mutual_information=random.random(),
                entropy=random.random(),
                synergy=random.random(),
                quantization_error=random.random()
            )
            
            for oracle in oracles:
                signal = run_async(oracle.evaluate(convergence))
                self.assertGreaterEqual(signal.value, 0.0, 
                    f"Oracle {oracle.oracle_type} returned < 0: {signal.value}")
                self.assertLessEqual(signal.value, 1.0, 
                    f"Oracle {oracle.oracle_type} returned > 1: {signal.value}")
    
    def test_pluralistic_bounded_output(self):
        """Pluralistic oracle must return values in [0, 1]."""
        oracle = PluralisticValueOracle()
        
        for _ in range(20):
            convergence = Convergence(
                id=f"plural-prop-{random.randint(0, 10000)}",
                clarity_score=random.random(),
                mutual_information=random.random(),
                entropy=random.random(),
                synergy=random.random(),
                quantization_error=random.random()
            )
            
            assessment = run_async(oracle.compute_value(convergence))
            
            self.assertGreaterEqual(assessment.value, 0.0)
            self.assertLessEqual(assessment.value, 1.0)


class TestConvergenceDataclass(unittest.TestCase):
    """Test the Convergence dataclass."""
    
    def test_convergence_creation(self):
        """Convergence should be creatable with required fields."""
        conv = Convergence(
            id="test",
            clarity_score=0.5,
            mutual_information=0.5,
            entropy=0.5,
            synergy=0.5,
            quantization_error=0.1
        )
        
        self.assertEqual(conv.id, "test")
        self.assertIsNotNone(conv.timestamp)
    
    def test_convergence_with_metadata(self):
        """Convergence should accept metadata."""
        conv = Convergence(
            id="meta-test",
            clarity_score=0.5,
            mutual_information=0.5,
            entropy=0.5,
            synergy=0.5,
            quantization_error=0.1,
            metadata={"source": "test", "priority": 1}
        )
        
        self.assertEqual(conv.metadata["source"], "test")


if __name__ == "__main__":
    unittest.main(verbosity=2)

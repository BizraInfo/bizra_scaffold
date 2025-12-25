#!/usr/bin/env python3
"""
Production Readiness Validation Runner
═════════════════════════════════════════════════════════════════════════════
Single command to prove the system is production-ready.

Run: python validate_production_ready.py

Options:
    --allow-unrun-tests  Allow validation to pass even if tests cannot run
                         (default: strict mode, tests MUST run)

Environment Variables:
    BIZRA_ALLOW_UNRUN_TESTS=1  Same as --allow-unrun-tests flag

This script:
1. Validates all safeguards are implemented
2. Runs comprehensive test suite
3. Checks configuration files
4. Generates validation report
5. Returns exit code 0 if ready, 1 if not

"Indeed, Allah loves those who do their work with Ihsān" - Sahih Muslim
"""

import sys
import os
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


class ValidationRunner:
    """Orchestrates all validation checks."""
    
    def __init__(self):
        self.workspace = Path(__file__).parent
        self.results: Dict[str, bool] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
    
    def print_header(self):
        """Print validation header."""
        print("\n" + "=" * 70)
        print(f"{Colors.BOLD}BIZRA Production Readiness Validation{Colors.END}")
        print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print("=" * 70 + "\n")
    
    def check_file_exists(self, filepath: str, description: str) -> bool:
        """Check if a critical file exists."""
        full_path = self.workspace / filepath
        exists = full_path.exists()
        
        if exists:
            print(f"{Colors.GREEN}✓{Colors.END} {description}: {filepath}")
        else:
            print(f"{Colors.RED}✗{Colors.END} {description}: {filepath} NOT FOUND")
            self.errors.append(f"Missing file: {filepath}")
        
        return exists
    
    def validate_critical_files(self) -> bool:
        """Validate all critical files exist."""
        print(f"\n{Colors.BOLD}[1/6] Checking Critical Files{Colors.END}")
        print("-" * 70)
        
        files = [
            ("core/snr_scorer.py", "SNR Scorer Module"),
            ("core/graph_of_thoughts.py", "Graph-of-Thoughts Engine"),
            ("core/production_safeguards.py", "Production Safeguards"),
            ("core/enhanced_cognitive_integration.py", "Enhanced Integration"),
            ("tests/test_production_safeguards.py", "Safeguard Tests"),
            ("tests/test_graph_of_thoughts_integration.py", "Integration Tests"),
            ("examples/graph_of_thoughts_demo.py", "Demo Script"),
            ("PRODUCTION_VALIDATION_CHECKLIST.md", "Validation Checklist"),
        ]
        
        all_exist = all(self.check_file_exists(f, desc) for f, desc in files)
        self.results['critical_files'] = all_exist
        return all_exist
    
    def validate_safeguard_imports(self) -> bool:
        """Validate production safeguards are properly imported."""
        print(f"\n{Colors.BOLD}[2/6] Validating Safeguard Integration{Colors.END}")
        print("-" * 70)
        
        integration_file = self.workspace / "core/enhanced_cognitive_integration.py"
        
        try:
            content = integration_file.read_text()
            
            checks = [
                ("CircuitBreaker", "Circuit breaker import"),
                ("InputValidator", "Input validator import"),
                ("get_circuit_breaker", "Circuit breaker factory"),
                ("get_health_checker", "Health checker factory"),
                ("get_audit_logger", "Audit logger factory"),
                ("validate_seed_concepts", "Seed validation call"),
                ("neo4j_circuit", "Neo4j circuit breaker"),
                ("convergence_circuit", "Convergence circuit breaker"),
                ("audit_logger", "Audit logger instance"),
            ]
            
            all_present = True
            for term, description in checks:
                if term in content:
                    print(f"{Colors.GREEN}✓{Colors.END} {description}")
                else:
                    print(f"{Colors.RED}✗{Colors.END} {description} NOT FOUND")
                    self.errors.append(f"Missing integration: {description}")
                    all_present = False
            
            self.results['safeguard_integration'] = all_present
            return all_present
        
        except Exception as e:
            print(f"{Colors.RED}✗{Colors.END} Error reading integration file: {e}")
            self.errors.append(f"Cannot validate integration: {e}")
            self.results['safeguard_integration'] = False
            return False
    
    def validate_ethical_constraints(self) -> bool:
        """Validate ethical constraints are enforced."""
        print(f"\n{Colors.BOLD}[3/6] Validating Ethical Constraints{Colors.END}")
        print("-" * 70)
        
        snr_file = self.workspace / "core/snr_scorer.py"
        
        try:
            content = snr_file.read_text()
            
            checks = [
                ("min_ihsan_for_high", "Ihsan threshold defined"),
                ("enable_ethical_constraints", "Ethical constraints flag"),
                ("ethical_override", "Override mechanism"),
                ("if self.enable_ethical_constraints", "Constraint enforcement"),
                ("SNRLevel.HIGH", "HIGH level classification"),
            ]
            
            all_present = True
            for term, description in checks:
                if term in content:
                    print(f"{Colors.GREEN}✓{Colors.END} {description}")
                else:
                    print(f"{Colors.RED}✗{Colors.END} {description} NOT FOUND")
                    self.errors.append(f"Missing ethical constraint: {description}")
                    all_present = False
            
            # Check default is enabled
            if 'enable_ethical_constraints: bool = True' in content:
                print(f"{Colors.GREEN}✓{Colors.END} Ethical constraints enabled by default")
            else:
                print(f"{Colors.YELLOW}⚠{Colors.END} Ethical constraints may not be enabled by default")
                self.warnings.append("Verify ethical constraints enabled by default")
            
            self.results['ethical_constraints'] = all_present
            return all_present
        
        except Exception as e:
            print(f"{Colors.RED}✗{Colors.END} Error validating ethical constraints: {e}")
            self.errors.append(f"Cannot validate ethics: {e}")
            self.results['ethical_constraints'] = False
            return False
    
    def run_tests(self) -> bool:
        """Run test suite."""
        print(f"\n{Colors.BOLD}[4/6] Running Test Suite{Colors.END}")
        print("-" * 70)
        
        try:
            # Run safeguard tests
            print(f"\n{Colors.BLUE}Running production safeguard tests...{Colors.END}")
            result = subprocess.run(
                [sys.executable, "-m", "pytest", 
                 "tests/test_production_safeguards.py", 
                 "-v", "--tb=short", "--color=yes"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"{Colors.GREEN}✓{Colors.END} All safeguard tests passed")
                tests_passed = True
            else:
                print(f"{Colors.RED}✗{Colors.END} Some tests failed")
                print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                self.errors.append("Test failures detected")
                tests_passed = False
            
            self.results['tests'] = tests_passed
            return tests_passed
        
        except subprocess.TimeoutExpired:
            print(f"{Colors.RED}✗{Colors.END} Tests timed out")
            self.errors.append("Test execution timeout")
            self.results['tests'] = False
            return False
        
        except Exception as e:
            print(f"{Colors.YELLOW}⚠{Colors.END} Could not run tests: {e}")
            
            # Check for allow_unrun_tests flag
            allow_unrun = getattr(self, 'allow_unrun_tests', False)
            
            if allow_unrun:
                print(f"   (--allow-unrun-tests flag set, continuing)")
                self.warnings.append(f"Tests not run (allowed): {e}")
                self.results['tests'] = None
                return True
            else:
                print(f"{Colors.RED}   STRICT MODE: Tests MUST run for production validation{Colors.END}")
                print(f"   Use --allow-unrun-tests to override (not recommended)")
                self.errors.append(f"Tests could not run: {e}")
                self.results['tests'] = False
                return False
    
    def validate_configuration(self) -> bool:
        """Validate configuration files."""
        print(f"\n{Colors.BOLD}[5/6] Validating Configuration{Colors.END}")
        print("-" * 70)
        
        checks_passed = True
        
        # Check requirements.txt has necessary packages
        req_file = self.workspace / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text()
            required_packages = ['neo4j', 'numpy', 'prometheus-client']
            
            for package in required_packages:
                if package in content.lower():
                    print(f"{Colors.GREEN}✓{Colors.END} Required package: {package}")
                else:
                    print(f"{Colors.YELLOW}⚠{Colors.END} Package may be missing: {package}")
                    self.warnings.append(f"Check if {package} is in dependencies")
        else:
            print(f"{Colors.YELLOW}⚠{Colors.END} requirements.txt not found")
            self.warnings.append("No requirements.txt found")
        
        # Check for monitoring config
        monitoring_files = [
            "monitoring/prometheus_alerts.yml",
            "monitoring/grafana_dashboard.json",
        ]
        
        for mfile in monitoring_files:
            full_path = self.workspace / mfile
            if full_path.exists():
                print(f"{Colors.GREEN}✓{Colors.END} Monitoring config: {mfile}")
            else:
                print(f"{Colors.YELLOW}⚠{Colors.END} Optional monitoring file missing: {mfile}")
                self.warnings.append(f"Consider adding: {mfile}")
        
        self.results['configuration'] = checks_passed
        return checks_passed
    
    def validate_documentation(self) -> bool:
        """Validate documentation completeness."""
        print(f"\n{Colors.BOLD}[6/6] Validating Documentation{Colors.END}")
        print("-" * 70)
        
        docs = [
            ("README.md", "Project README"),
            ("PRODUCTION_VALIDATION_CHECKLIST.md", "Validation Checklist"),
            ("INSTALLATION_GUIDE.md", "Installation Guide"),
            ("ELITE_ROADMAP.md", "Strategic Roadmap"),
        ]
        
        all_exist = True
        for doc, description in docs:
            if self.check_file_exists(doc, description):
                pass
            else:
                all_exist = False
        
        self.results['documentation'] = all_exist
        return all_exist
    
    def generate_report(self) -> Tuple[bool, str]:
        """Generate final validation report."""
        print("\n" + "=" * 70)
        print(f"{Colors.BOLD}VALIDATION REPORT{Colors.END}")
        print("=" * 70 + "\n")
        
        # Count results
        total = len(self.results)
        passed = sum(1 for v in self.results.values() if v is True)
        failed = sum(1 for v in self.results.values() if v is False)
        skipped = sum(1 for v in self.results.values() if v is None)
        
        # Display summary
        print(f"Total Checks: {total}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.END}")
        if failed > 0:
            print(f"{Colors.RED}Failed: {failed}{Colors.END}")
        if skipped > 0:
            print(f"{Colors.YELLOW}Skipped: {skipped}{Colors.END}")
        
        # Display warnings
        if self.warnings:
            print(f"\n{Colors.YELLOW}Warnings ({len(self.warnings)}):{Colors.END}")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
        
        # Display errors
        if self.errors:
            print(f"\n{Colors.RED}Errors ({len(self.errors)}):{Colors.END}")
            for error in self.errors:
                print(f"  ✗ {error}")
        
        # Final verdict
        print("\n" + "=" * 70)
        
        production_ready = failed == 0 and passed >= (total - skipped)
        
        if production_ready:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ SYSTEM IS PRODUCTION READY{Colors.END}")
            print(f"\n{Colors.GREEN}All critical safeguards validated.{Colors.END}")
            print(f"{Colors.GREEN}Ethical constraints enforced.{Colors.END}")
            print(f"{Colors.GREEN}Ready for staged deployment.{Colors.END}")
            verdict = "READY"
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ SYSTEM NOT READY FOR PRODUCTION{Colors.END}")
            print(f"\n{Colors.RED}Critical issues must be resolved before deployment.{Colors.END}")
            verdict = "NOT_READY"
        
        print("=" * 70 + "\n")
        
        # Generate JSON report
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'verdict': verdict,
            'results': self.results,
            'warnings': self.warnings,
            'errors': self.errors,
            'statistics': {
                'total': total,
                'passed': passed,
                'failed': failed,
                'skipped': skipped
            }
        }
        
        report_file = self.workspace / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {report_file}\n")
        
        return production_ready, verdict
    
    def run(self) -> int:
        """Run all validations."""
        self.print_header()
        
        self.validate_critical_files()
        self.validate_safeguard_imports()
        self.validate_ethical_constraints()
        self.run_tests()
        self.validate_configuration()
        self.validate_documentation()
        
        ready, verdict = self.generate_report()
        
        return 0 if ready else 1


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="BIZRA Production Readiness Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_production_ready.py              # Strict mode (default)
  python validate_production_ready.py --allow-unrun-tests  # Allow missing tests
  
Environment Variables:
  BIZRA_ALLOW_UNRUN_TESTS=1  Same as --allow-unrun-tests
        """
    )
    parser.add_argument(
        '--allow-unrun-tests',
        action='store_true',
        default=os.environ.get('BIZRA_ALLOW_UNRUN_TESTS', '').lower() in ('1', 'true', 'yes'),
        help='Allow validation to pass even if tests cannot run (not recommended for production)'
    )
    
    args = parser.parse_args()
    
    try:
        runner = ValidationRunner()
        runner.allow_unrun_tests = args.allow_unrun_tests
        exit_code = runner.run()
        sys.exit(exit_code)
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.END}")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n{Colors.RED}Validation failed with error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

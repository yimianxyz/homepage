"""
Test Runner - Run all RL system tests

This script runs all test suites and provides a comprehensive
test report for the RL training system.
"""

import sys
import os
import argparse
import time
from typing import List, Tuple

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.utils import set_seed


def run_test_suite(test_module_name: str, test_function_name: str) -> Tuple[bool, float]:
    """
    Run a specific test suite
    
    Args:
        test_module_name: Name of the test module
        test_function_name: Name of the test function to run
        
    Returns:
        success: Whether all tests passed
        duration: Test duration in seconds
    """
    print(f"\n🔄 Running {test_module_name}...")
    
    start_time = time.time()
    
    try:
        # Import the test module dynamically with full path
        full_module_name = f"rl.tests.{test_module_name}"
        test_module = __import__(full_module_name, fromlist=[test_function_name])
        test_function = getattr(test_module, test_function_name)
        
        # Run the test function
        success = test_function()
        
        duration = time.time() - start_time
        
        if success:
            print(f"✅ {test_module_name} completed successfully ({duration:.1f}s)")
        else:
            print(f"❌ {test_module_name} failed ({duration:.1f}s)")
        
        return success, duration
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 {test_module_name} crashed: {e} ({duration:.1f}s)")
        return False, duration


def run_all_tests(quick: bool = False) -> None:
    """
    Run all test suites
    
    Args:
        quick: Whether to run quick tests only
    """
    print("🚀 STARTING RL SYSTEM TEST SUITE")
    print("=" * 60)
    print(f"Quick mode: {quick}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Define test suites
    test_suites = [
        ("test_environment", "run_all_environment_tests"),
        ("test_model_loading", "run_all_model_tests"),
    ]
    
    # Add integration tests if not in quick mode
    if not quick:
        test_suites.append(("test_integration", "run_all_integration_tests"))
    
    # Run test suites
    results = []
    total_duration = 0.0
    
    for test_module, test_function in test_suites:
        success, duration = run_test_suite(test_module, test_function)
        results.append((test_module, success, duration))
        total_duration += duration
        
        # Stop on first failure if requested
        if not success and quick:
            print(f"\n⏹️  Stopping on first failure (quick mode)")
            break
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST REPORT")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_module, success, duration in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_module:<20} {status} ({duration:.1f}s)")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    total_tests = passed + failed
    success_rate = passed / total_tests if total_tests > 0 else 0
    
    print(f"\n📈 Summary:")
    print(f"  Total test suites: {total_tests}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Total time: {total_duration:.1f}s")
    
    if passed == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("✨ The RL system is ready for training!")
    else:
        print(f"\n⚠️  {failed} TEST SUITE(S) FAILED")
        print("🔧 Please fix the issues before running training")
    
    return passed == total_tests


def check_dependencies() -> bool:
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch',
        'numpy', 
        'gymnasium',
        'stable_baselines3',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages before running tests")
        return False
    
    print("✅ All dependencies available")
    return True


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Run RL system tests")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick tests only (skip integration tests)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--check-deps", action="store_true",
                        help="Check dependencies only")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check dependencies
    if args.check_deps:
        deps_ok = check_dependencies()
        exit(0 if deps_ok else 1)
    
    # Check dependencies before running tests
    if not check_dependencies():
        exit(1)
    
    # Run tests
    try:
        success = run_all_tests(quick=args.quick)
        exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
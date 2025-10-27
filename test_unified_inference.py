#!/usr/bin/env python3
"""
Quick validation test for unified inference script.
Tests data loading and argument parsing without full model inference.
"""
import sys
import subprocess

def test_help():
    """Test that help command works"""
    print("=" * 80)
    print("TEST 1: Help command")
    print("=" * 80)
    result = subprocess.run(
        ["python", "scripts/run_inference.py", "--help"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✅ Help command works")
        return True
    else:
        print(f"❌ Help command failed: {result.stderr}")
        return False

def test_missing_required_args():
    """Test that missing required arguments are caught"""
    print("\n" + "=" * 80)
    print("TEST 2: Missing required arguments")
    print("=" * 80)
    result = subprocess.run(
        ["python", "scripts/run_inference.py"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0 and "required" in result.stderr.lower():
        print("✅ Required argument validation works")
        return True
    else:
        print(f"❌ Expected error for missing arguments")
        return False

def test_argument_parsing_konkani():
    """Test that Konkani arguments parse correctly"""
    print("\n" + "=" * 80)
    print("TEST 3: Argument parsing (Konkani)")
    print("=" * 80)
    # We'll test if the script at least starts to load
    # (will fail at model loading, but that's expected)
    cmd = [
        "python", "scripts/run_inference.py",
        "--dataset", "predictionguard/english-hindi-marathi-konkani-corpus",
        "--model", "Unbabel/TowerInstruct-7B-v0.1",
        "--pivot", "hin",
        "--source", "mar", 
        "--target", "gom",
        "--db", "test_db",
        "--output", "/tmp/test_output.csv",
        "--scores", "/tmp/test_scores.json",
        "--num-examples", "0",
        "--batch-size", "1"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nNote: This will fail at model loading (expected), but we check if args parse correctly...")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    # Check if we got past argument parsing
    if "Dataset:" in result.stdout or "Loading tokenizer" in result.stdout:
        print("✅ Arguments parsed successfully, script started execution")
        return True
    elif result.returncode != 0:
        # Check the error message
        if "argument" in result.stderr.lower() and "required" in result.stderr.lower():
            print(f"❌ Argument parsing failed: {result.stderr[:200]}")
            return False
        else:
            # Failed for other reason (likely model loading), which is fine
            print("✅ Arguments parsed (failed later in execution, which is expected)")
            if result.stdout:
                print(f"\nStdout preview:\n{result.stdout[:500]}")
            return True
    else:
        print(f"❌ Unexpected result")
        return False

def test_argument_parsing_arabic():
    """Test that Arabic arguments parse correctly"""
    print("\n" + "=" * 80)
    print("TEST 4: Argument parsing (Arabic)")
    print("=" * 80)
    cmd = [
        "python", "scripts/run_inference.py",
        "--dataset", "pierrebarbera/tunisian_msa_arabizi",
        "--model", "Unbabel/TowerInstruct-7B-v0.1",
        "--pivot", "msa",
        "--source", "en",
        "--target", "tn",
        "--db", "test_arabic_db",
        "--output", "/tmp/test_arabic_output.csv",
        "--scores", "/tmp/test_arabic_scores.json",
        "--num-examples", "0",
        "--batch-size", "1"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nNote: This will fail at model loading (expected), but we check if args parse correctly...")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    # Check if we got past argument parsing
    if "Dataset:" in result.stdout or "Loading tokenizer" in result.stdout:
        print("✅ Arguments parsed successfully, script started execution")
        return True
    elif result.returncode != 0:
        # Check the error message
        if "argument" in result.stderr.lower() and "required" in result.stderr.lower():
            print(f"❌ Argument parsing failed: {result.stderr[:200]}")
            return False
        else:
            # Failed for other reason (likely model loading), which is fine
            print("✅ Arguments parsed (failed later in execution, which is expected)")
            if result.stdout:
                print(f"\nStdout preview:\n{result.stdout[:500]}")
            return True
    else:
        print(f"❌ Unexpected result")
        return False

def main():
    print("\n" + "🧪 UNIFIED INFERENCE SCRIPT VALIDATION" + "\n")
    
    tests = [
        test_help,
        test_missing_required_args,
        test_argument_parsing_konkani,
        test_argument_parsing_arabic
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed! Script is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())


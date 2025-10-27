#!/usr/bin/env python3
"""
Quick validation test for run_inference.py script.
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
        ["conda", "run", "-n", "lrlt_exp", "python", "scripts/run_inference.py", "--help"],
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
        ["conda", "run", "-n", "lrlt_exp", "python", "scripts/run_inference.py"],
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
    """Test that Konkani inference runs successfully (3 samples, zero-shot)"""
    print("\n" + "=" * 80)
    print("TEST 3: Full Inference Test (Konkani, 3 samples)")
    print("=" * 80)
    cmd = [
        "conda", "run", "-n", "lrlt_exp",
        "python", "scripts/run_inference.py",
        "--dataset", "predictionguard/english-hindi-marathi-konkani-corpus",
        "--model", "Unbabel/TowerInstruct-7B-v0.1",
        "--pivot", "hin",
        "--source", "mar", 
        "--target", "gom",
        "--db", "translations_db",  # Use existing DB
        "--output", "/tmp/test_konkani_output.csv",
        "--scores", "/tmp/test_konkani_scores.json",
        "--num-examples", "0",  # Zero-shot
        "--batch-size", "1",
        "--test-limit", "3"  # Only process 3 samples
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nNote: Will load model and process 3 test samples (may take 2-3 minutes)...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180  # 3 minutes for model loading + inference
        )
        
        # Check if inference completed successfully
        if result.returncode == 0 and "Inference completed" in result.stdout:
            print("✅ Inference completed successfully!")
            # Show some output
            lines = result.stdout.split('\n')
            for line in lines:
                if any(x in line for x in ["BLEU", "chrF", "✅", "Test samples:"]):
                    print(f"   {line.strip()}")
            return True
        elif result.returncode == 0:
            print("✅ Script ran but couldn't verify completion (check output)")
            print(f"\nLast 300 chars of stdout:\n{result.stdout[-300:]}")
            return True
        elif "argument" in result.stderr.lower() and "required" in result.stderr.lower():
            print(f"❌ Argument parsing failed: {result.stderr[:200]}")
            return False
        else:
            print(f"❌ Script failed with error")
            print(f"Exit code: {result.returncode}")
            print(f"Stderr preview:\n{result.stderr[:300]}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Test timed out after 3 minutes (model may be downloading)")
        print("   This could be normal for first run. Try running manually.")
        return False

def test_argument_parsing_arabic():
    """Test that Arabic inference runs successfully (3 samples, zero-shot)"""
    print("\n" + "=" * 80)
    print("TEST 4: Full Inference Test (Arabic, 3 samples)")
    print("=" * 80)
    cmd = [
        "conda", "run", "-n", "lrlt_exp",
        "python", "scripts/run_inference.py",
        "--dataset", "predictionguard/arabic_acl_corpus",
        "--model", "Unbabel/TowerInstruct-7B-v0.1",
        "--pivot", "msa",
        "--source", "en",
        "--target", "tn",
        "--db", "arabic_translations",  # Use existing DB
        "--output", "/tmp/test_arabic_output.csv",
        "--scores", "/tmp/test_arabic_scores.json",
        "--num-examples", "0",  # Zero-shot
        "--batch-size", "1",
        "--test-limit", "3"  # Only process 3 samples
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nNote: Will load model and process 3 test samples (model already loaded from Konkani test)...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180  # 3 minutes for model loading + inference
        )
        
        # Check if inference completed successfully
        if result.returncode == 0 and "Inference completed" in result.stdout:
            print("✅ Inference completed successfully!")
            # Show some output
            lines = result.stdout.split('\n')
            for line in lines:
                if any(x in line for x in ["BLEU", "chrF", "✅", "Test samples:"]):
                    print(f"   {line.strip()}")
            return True
        elif result.returncode == 0:
            print("✅ Script ran but couldn't verify completion (check output)")
            print(f"\nLast 300 chars of stdout:\n{result.stdout[-300:]}")
            return True
        elif "argument" in result.stderr.lower() and "required" in result.stderr.lower():
            print(f"❌ Argument parsing failed: {result.stderr[:200]}")
            return False
        else:
            print(f"❌ Script failed with error")
            print(f"Exit code: {result.returncode}")
            print(f"Stderr preview:\n{result.stderr[:300]}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Test timed out after 3 minutes (model may be downloading)")
        print("   This could be normal for first run. Try running manually.")
        return False

def main():
    print("\n" + "🧪 INFERENCE SCRIPT VALIDATION (run_inference.py)" + "\n")
    
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


#!/usr/bin/env python3
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    logger.info("Testing imports...")
    try:
        from data.name_utils import normalize_name, fuzzy_join_names
        logger.info("✓ name_utils imports")
    except Exception as e:
        logger.error(f"✗ name_utils: {e}")
        return False

    try:
        from data.build_training_dataset import DatasetBuilder
        logger.info("✓ build_training_dataset imports")
    except Exception as e:
        logger.error(f"✗ build_training_dataset: {e}")
        return False

    try:
        from data.feature_engineering import engineer_features, PARK_FACTORS
        logger.info("✓ feature_engineering imports")
    except Exception as e:
        logger.error(f"✗ feature_engineering: {e}")
        return False

    try:
        from models.train_models_v4 import ModelPipeline
        logger.info("✓ train_models_v4 imports")
    except Exception as e:
        logger.error(f"✗ train_models_v4: {e}")
        return False

    return True


def test_database_connection():
    logger.info("\nTesting database connection...")
    try:
        from data.build_training_dataset import DatasetBuilder
        builder = DatasetBuilder()
        if builder.conn is None:
            logger.error("✗ Database connection failed")
            return False
        builder.close()
        logger.info("✓ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Database connection error: {e}")
        return False


def test_park_factors():
    logger.info("\nTesting park factors...")
    try:
        from data.feature_engineering import PARK_FACTORS
        expected_teams = ['COL', 'NYY', 'BOS', 'OAK', 'TBR']
        for team in expected_teams:
            if team not in PARK_FACTORS:
                logger.error(f"✗ Missing park factor for {team}")
                return False
        logger.info(f"✓ Park factors loaded: {len(PARK_FACTORS)} teams")
        return True
    except Exception as e:
        logger.error(f"✗ Park factors error: {e}")
        return False


def test_directories():
    logger.info("\nTesting required directories...")
    dirs_to_check = [
        Path(__file__).parent / 'models' / 'artifacts',
        Path(__file__).parent / 'output',
        Path(__file__).parent / 'data'
    ]
    for d in dirs_to_check:
        if not d.exists():
            logger.error(f"✗ Missing directory: {d}")
            return False
        logger.info(f"✓ Directory exists: {d.name}")
    return True


def test_name_normalization():
    logger.info("\nTesting name normalization...")
    try:
        from data.name_utils import normalize_name

        test_cases = [
            ("Aaron Judge", "aaron judge"),
            ("Judge, Aaron", "aaron judge"),
            ("José Ramírez", "jose ramirez"),
            ("Juan Soto Jr.", "juan soto"),
            ("Randy Johnson Sr.", "randy johnson"),
        ]

        for input_name, expected in test_cases:
            result = normalize_name(input_name)
            if result != expected:
                logger.error(f"✗ {input_name} → {result} (expected {expected})")
                return False

        logger.info("✓ Name normalization working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Name normalization error: {e}")
        return False


def run_all_tests():
    logger.info("=" * 80)
    logger.info("ProjectionAI Integration Test Suite")
    logger.info("=" * 80)

    tests = [
        ("Imports", test_imports),
        ("Park Factors", test_park_factors),
        ("Directories", test_directories),
        ("Name Normalization", test_name_normalization),
        ("Database Connection", test_database_connection),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"✗ {name} failed with exception: {e}")
            results.append((name, False))

    logger.info("\n" + "=" * 80)
    logger.info("Test Results Summary")
    logger.info("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status:8} - {name}")

    logger.info("=" * 80)
    logger.info(f"Total: {passed}/{total} tests passed")
    logger.info("=" * 80)

    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

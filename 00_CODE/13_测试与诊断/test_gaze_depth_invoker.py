"""Strict common-cohort contract for GazeDepth."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from INDICATORS_SIGNIFICANCE.TEST.test_evacuation_invokers import ContractChecks


class InvokerTests(ContractChecks, unittest.TestCase):
    modules = ['GazeDepth']


if __name__ == '__main__':
    unittest.main()

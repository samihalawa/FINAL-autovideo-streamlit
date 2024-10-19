import unittest
from autocoder4 import validate_api_key, run_tests, run_linter

class TestAutocoder4(unittest.TestCase):
    def test_validate_api_key(self):
        self.assertTrue(validate_api_key("sk-" + "a" * 48))
        self.assertFalse(validate_api_key("invalid-key"))

    def test_run_tests(self):
        script = "def test_function():\n    assert True"
        result = run_tests(script)
        self.assertIn("1 passed", result)

    def test_run_linter(self):
        script = "def test_function():\n    x = 1\n    y = 2\n    return x + y"
        result = run_linter(script)
        self.assertEqual(result, "")  # No linting errors

if __name__ == '__main__':
    unittest.main()

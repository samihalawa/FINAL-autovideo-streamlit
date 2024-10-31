import unittest
from unittest.mock import patch, MagicMock
from autocoder4 import validate_api_key, extract_sections, optimize_script
import asyncio
import ast

class TestAutocoder4(unittest.TestCase):
    """Test suite for Autocoder4 functionality"""

    def setUp(self):
        """Set up test environment"""
        self.valid_api_key = "sk-" + "a" * 48
        self.invalid_api_key = "invalid-key"
        self.test_script = """
def test_function():
    x = 1
    y = 2
    return x + y
"""

    def test_validate_api_key(self):
        """Test API key validation"""
        self.assertTrue(validate_api_key(self.valid_api_key))
        self.assertFalse(validate_api_key(self.invalid_api_key))
        self.assertFalse(validate_api_key(""))
        self.assertFalse(validate_api_key(None))

    def test_extract_sections(self):
        """Test code section extraction"""
        sections = extract_sections(self.test_script)
        self.assertIn("test_function", sections)
        self.assertEqual(len(sections), 1)

        # Test with invalid input
        self.assertEqual(extract_sections("invalid python code{"), {"main": "invalid python code{"})

    @patch('openai.ChatCompletion.acreate')
    async def test_optimize_script(self, mock_acreate):
        """Test script optimization"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="optimized code"))]
        mock_acreate.return_value = mock_response

        result = await optimize_script(self.test_script)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_syntax_validation(self):
        """Test Python syntax validation"""
        valid_code = "x = 1\ny = 2\nprint(x + y)"
        invalid_code = "x = 1\ny = \nprint(x + y"

        try:
            ast.parse(valid_code)
            valid = True
        except SyntaxError:
            valid = False
        self.assertTrue(valid)

        try:
            ast.parse(invalid_code)
            valid = True
        except SyntaxError:
            valid = False
        self.assertFalse(valid)

def run_async_tests():
    """Run async tests with proper event loop handling"""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(TestAutocoder4().test_optimize_script())

if __name__ == '__main__':
    unittest.main()

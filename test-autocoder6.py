import unittest
from unittest.mock import patch, MagicMock
import ast
from autocoder6 import (
    segment_code,
    DependencyCheckerAgent,
    extract_sections,
    optimize_with_aider,
    validate_api_key
)

class TestAutocoder6(unittest.TestCase):
    """Test suite for Autocoder6 functionality"""

    def setUp(self):
        """Set up test environment"""
        self.test_code = '''
import os
import sys

def foo():
    print(sys.version)
    return os.name

class Bar:
    def method(self):
        print("Hello")

print("Hello World")
'''
        self.dependency_checker = DependencyCheckerAgent()

    def test_segment_code_valid_input(self):
        """Test code segmentation with valid input"""
        segments = segment_code(self.test_code)
        self.assertIn('Imports', segments)
        self.assertIn('def foo', segments)
        self.assertIn('class Bar', segments)
        self.assertIn('Global', segments)

    def test_segment_code_syntax_error(self):
        """Test code segmentation with invalid input"""
        invalid_code = 'def invalid_code('
        segments = segment_code(invalid_code)
        self.assertEqual(segments, {})

    def test_dependency_checker_unused_imports(self):
        """Test dependency checker for unused imports"""
        code_blocks = {
            'Imports': 'import os\nimport sys\n',
            'def foo': 'def foo():\n    print(sys.version)\n',
            'Global': 'print("Hello World")\n'
        }
        unused_imports = self.dependency_checker.check_dependencies(code_blocks)
        self.assertIn('os', unused_imports)

    def test_dependency_checker_all_used(self):
        """Test dependency checker when all imports are used"""
        code_blocks = {
            'Imports': 'import os\nimport sys\n',
            'def foo': 'def foo():\n    print(os.name)\n    print(sys.version)\n',
        }
        unused_imports = self.dependency_checker.check_dependencies(code_blocks)
        self.assertEqual(len(unused_imports), 0)

    @patch('aider.chat.Chat')
    def test_optimize_with_aider(self, mock_chat):
        """Test code optimization with Aider"""
        mock_chat.return_value.send_message.return_value = MagicMock(content="optimized code")
        result = optimize_with_aider(self.test_code)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_extract_sections(self):
        """Test section extraction from code"""
        sections = extract_sections(self.test_code)
        self.assertIn('imports', sections)
        self.assertIn('function_definitions', sections)
        self.assertIn('class_definitions', sections)

    def test_validate_api_key(self):
        """Test API key validation"""
        valid_key = "sk-" + "a" * 48
        invalid_key = "invalid-key"
        
        self.assertTrue(validate_api_key(valid_key))
        self.assertFalse(validate_api_key(invalid_key))
        self.assertFalse(validate_api_key(""))
        self.assertFalse(validate_api_key(None))

if __name__ == '__main__':
    unittest.main()

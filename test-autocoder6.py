
import unittest
from code_enhancer import segment_code, DependencyCheckerAgent

class TestCodeEnhancer(unittest.TestCase):

    def test_segment_code_valid_input(self):
        code = '''
import os
import sys

def foo():
    pass

class Bar:
    def method(self):
        pass

print("Hello World")
'''
        segments = segment_code(code)
        self.assertIn('Imports', segments)
        self.assertIn('def foo', segments)
        self.assertIn('class Bar', segments)
        self.assertIn('Global', segments)

    def test_segment_code_syntax_error(self):
        code = 'def invalid_code('  # Missing closing parenthesis
        segments = segment_code(code)
        self.assertEqual(segments, {})  # Should return an empty dict on syntax error

    def test_dependency_checker_unused_imports(self):
        code_blocks = {
            'Imports': 'import os\nimport sys\n',
            'def foo': 'def foo():\n    print(sys.version)\n',
            'Global': 'print("Hello World")\n'
        }
        agent = DependencyCheckerAgent()
        unused_imports = agent.check_dependencies(code_blocks)
        self.assertEqual(unused_imports, {'os'})

    def test_dependency_checker_all_used(self):
        code_blocks = {
            'Imports': 'import os\nimport sys\n',
            'def foo': 'def foo():\n    print(os.name)\n    print(sys.version)\n',
        }
        agent = DependencyCheckerAgent()
        unused_imports = agent.check_dependencies(code_blocks)
        self.assertEqual(unused_imports, set())

if __name__ == '__main__':
    unittest.main()

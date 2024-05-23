import unittest
from src.prompt.promptgenerator import PromptFactory

class TestPromptFactory(unittest.TestCase):
    def test_composite_docstring_prompt(self):
        factory = PromptFactory()
        API_description = "This API performs data aggregation."
        func_inputs = "input_data: DataFrame, columns: List[str]"
        func_outputs = "DataFrame"
        description_text = "Aggregates data based on specified columns."

        expected_prompt = f"""
Write a concise docstring for an invisible function in Python, focusing solely on its core functionality derived from the sequential composition of sub APIs.
- API Description: This API performs data aggregation.
- Parameters: input_data: DataFrame, columns: List[str]
- Returns: DataFrame
- Additional Description: Aggregates data based on specified columns.
Craft a 1-2 sentence docstring that extracts and polishes the core information. The response should be in reStructuredText format, excluding specific API names and unprofessional terms. Remember to use parameter details only to refine the core functionality explanation, not for plain input/output information.
"""

        prompt = factory.create_prompt('composite_docstring', API_description, func_inputs, func_outputs, description_text)
        self.assertEqual(prompt.strip(), expected_prompt.strip())

if __name__ == '__main__':
    unittest.main()

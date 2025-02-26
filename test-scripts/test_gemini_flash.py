# test-scripts/test_gemini_flash.py
import asyncio
from unittest import TestCase
from utils.answer_checker import AnswerChecker

class TestGeminiFlash(TestCase):
    def setUp(self):
        self.checker = AnswerChecker(no_think_tags=False)
        self.test_cases = [
            {
                "name": "Simple Addition",
                "question": "What is 2 + 2?",
                "model_answer": "<think>This is a simple addition problem. 2 plus 2 equals 4.</think> #### 4",
                "ground_truth": "4",
                "expected": True
            },
            {
                "name": "Complex Multi-step",
                "question": "A store has 120 apples. They sell 40 apples on Monday and 35 on Tuesday. How many apples remain?",
                "model_answer": """<think>
                    1. Start with total apples: 120
                    2. Subtract Monday sales: 120 - 40 = 80
                    3. Subtract Tuesday sales: 80 - 35 = 45
                    </think> #### 45""",
                "ground_truth": "45",
                "expected": True
            },
            {
                "name": "Incorrect Answer",
                "question": "What is 15 - 7?",
                "model_answer": "<think>15 minus 7 equals 9</think> #### 9",
                "ground_truth": "8",
                "expected": False
            }
        ]

    async def _run_test_case(self, case):
        result = await self.checker.check_answer_async(
            case["question"],
            case["model_answer"],
            case["ground_truth"]
        )
        return result["is_correct"] == case["expected"]

    async def run_tests(self):
        results = {}
        for case in self.test_cases:
            passed = await self._run_test_case(case)
            results[case["name"]] = passed
            print(f"Test '{case['name']}': {'PASSED' if passed else 'FAILED'}")
        
        print("\nTest Summary:")
        for name, passed in results.items():
            print(f"{name}: {'✅' if passed else '❌'}")
        
        print("\nChecker Stats:")
        for stat, value in self.checker.stats.items():
            print(f"{stat}: {value}")

def main():
    tester = TestGeminiFlash()
    asyncio.run(tester.run_tests())

if __name__ == "__main__":
    main()
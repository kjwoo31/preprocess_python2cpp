import ast
import unittest

from core.analysis.inferencer import TypeInferenceEngine
from core.analysis.parser import PythonASTParser
from core.intermediate.builder import IRBuilder
from core.intermediate.schema import IRPipeline


class TestIRBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = IRBuilder()
        self.parser = PythonASTParser()
        self.type_engine = TypeInferenceEngine()

    def test_simple_assignment(self):
        code = """
def test_func():
    a = 1
    b = 2
    return a
"""
        self.parser.parse(code)
        tree = ast.parse(code)
        pipeline = self.builder.build_pipeline(
            self.parser, "test_func", tree, self.type_engine
        )

        self.assertIsInstance(pipeline, IRPipeline)
        self.assertEqual(pipeline.name, "test_func")
        self.assertEqual(len(pipeline.operations), 2)
        self.assertEqual(pipeline.operations[0].output, "a")
        self.assertEqual(pipeline.operations[1].output, "b")

    def test_function_call(self):
        code = """
import numpy as np
def test_func():
    a = np.zeros((10, 10))
    return a
"""
        self.parser.parse(code)
        tree = ast.parse(code)
        pipeline = self.builder.build_pipeline(
            self.parser, "test_func", tree, self.type_engine
        )

        self.assertEqual(len(pipeline.operations), 1)
        op = pipeline.operations[0]
        self.assertEqual(op.op_type.value, "function_call")
        self.assertEqual(op.function, "zeros")
        self.assertEqual(op.source_lib, "np")


if __name__ == "__main__":
    unittest.main()

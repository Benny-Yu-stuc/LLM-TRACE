import unittest

from llm_trace.demo_data import build_demo_document
from llm_trace.pipeline import LlmTracePipeline


class DemoPipelineTest(unittest.TestCase):
    def test_demo_pipeline_runs_end_to_end(self):
        config = {
            "chief_engineer": {"cache_enabled": False, "api_key": "", "api_url": ""},
            "fusion": {"vector_dim": 16, "routing_bias": 0.15, "minimum_prior": 0.05},
            "extraction": {"entity_confidence_threshold": 0.35, "relation_confidence_threshold": 0.35},
            "graph": {},
        }
        pipeline = LlmTracePipeline(config)
        result = pipeline.run_document(build_demo_document())

        self.assertTrue(result.evidence)
        self.assertTrue(result.chief_engineer_outputs)
        self.assertTrue(result.fused_operations)
        self.assertTrue(result.entities)
        self.assertTrue(result.relations)
        self.assertTrue(result.graph.nodes)
        self.assertEqual(result.metrics["traceability"]["entity_traceability_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

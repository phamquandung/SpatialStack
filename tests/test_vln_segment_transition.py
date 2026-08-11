import unittest

import torch

from qwen_vl.model.vggt.eviction.vln_segment_transition import (
    VLNGhostTokenMetadata,
    compute_candidate_final_score,
    gather_metadata,
    zscore_normalize,
)


class ZScoreNormalizeTest(unittest.TestCase):
    def test_zero_mean_unit_population_variance_and_shape(self):
        scores = torch.tensor([[3.0, 1.0], [4.0, 2.0]])

        actual = zscore_normalize(scores)

        torch.testing.assert_close(actual.mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            actual.square().mean(), torch.tensor(1.0), atol=1e-6, rtol=0
        )
        self.assertEqual(actual.shape, scores.shape)
        self.assertEqual(actual.dtype, torch.float32)

    def test_constant_scores_are_neutral(self):
        actual = zscore_normalize(torch.full((5,), 7.0))

        torch.testing.assert_close(actual, torch.zeros(5))

    def test_rejects_empty_scores(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            zscore_normalize(torch.empty(0))


class CandidateFinalScoreTest(unittest.TestCase):
    @staticmethod
    def metadata(confidence):
        confidence = torch.tensor(confidence, dtype=torch.float32)
        n = confidence.numel()
        return VLNGhostTokenMetadata(
            frame_id=torch.arange(n, dtype=torch.int32),
            geometry_score=torch.tensor([0.0, 1.0, 3.0, 2.0, 0.0]),
            confidence_score=confidence,
            instruction_score=torch.tensor([0.0, 0.1, 0.4, 0.2, 0.0]),
            transition_score=torch.tensor([0.0, 0.4, 0.2, 0.9, 0.0]),
            final_score=torch.zeros(n),
            is_special=torch.tensor([True, False, False, False, True]),
            best_segment_id=torch.full((n,), -1, dtype=torch.int16),
        )

    def test_is_invariant_to_positive_affine_component_scale(self):
        weights = dict(geometry=0.3, confidence=0.2, instruction=0.3, transition=0.2)
        original, _ = compute_candidate_final_score(
            self.metadata([0.0, 1.0, 2.0, 4.0, 0.0]), weights, 0.3
        )
        rescaled, _ = compute_candidate_final_score(
            self.metadata([0.0, 1010.0, 1020.0, 1040.0, 0.0]), weights, 0.3
        )

        torch.testing.assert_close(original, rescaled)

    def test_special_tokens_score_above_every_patch(self):
        weights = dict(geometry=0.3, confidence=0.2, instruction=0.3, transition=0.2)
        metadata = self.metadata([0.0, 1.0, 2.0, 4.0, 0.0])

        final, normalized = compute_candidate_final_score(metadata, weights, 0.3)

        self.assertGreater(final[metadata.is_special].min(), final[~metadata.is_special].max())
        for values in normalized.values():
            torch.testing.assert_close(
                values.square().mean(), torch.tensor(1.0), atol=1e-6, rtol=0
            )

    def test_realistic_scale_mismatch_prunes_kv_and_metadata_together(self):
        tokens_per_frame = 205
        frames = 80
        total = tokens_per_frame * frames
        special = torch.zeros(total, dtype=torch.bool)
        special.reshape(frames, tokens_per_frame)[:, :5] = True
        patch = ~special
        patch_index = torch.arange(patch.sum(), dtype=torch.float32)
        frame_ids = torch.arange(frames, dtype=torch.int32).repeat_interleave(tokens_per_frame)

        geometry = frame_ids.float().mul(1e-4).add(0.5)
        confidence = torch.zeros(total, dtype=torch.float32)
        confidence[patch] = (patch_index.remainder(7) == 0).float().mul(0.4).add(0.55)
        instruction = torch.zeros(total, dtype=torch.float32)
        instruction[patch] = 0.5001 + patch_index.remainder(101).mul(1e-6)
        transition = torch.zeros(total, dtype=torch.float32)
        transition[patch] = frame_ids[patch].float().remainder(9).mul(1e-5)
        metadata = VLNGhostTokenMetadata(
            frame_id=frame_ids,
            geometry_score=geometry,
            confidence_score=confidence,
            instruction_score=instruction,
            transition_score=transition,
            final_score=torch.zeros(total, dtype=torch.float32),
            is_special=special,
            best_segment_id=torch.zeros(total, dtype=torch.int16),
        )
        weights = dict(geometry=0.3, confidence=0.2, instruction=0.3, transition=0.2)

        final, normalized = compute_candidate_final_score(metadata, weights, 0.3)

        expected_share = {
            "geometry": 9.0 / 26.0,
            "confidence": 4.0 / 26.0,
            "instruction": 9.0 / 26.0,
            "transition": 4.0 / 26.0,
        }
        variance_proxy = {
            name: (weights[name] * values.std(unbiased=False)).square()
            for name, values in normalized.items()
        }
        proxy_total = sum(variance_proxy.values())
        for name, values in normalized.items():
            torch.testing.assert_close(
                values.std(unbiased=False), torch.tensor(1.0), atol=1e-5, rtol=0
            )
            torch.testing.assert_close(
                variance_proxy[name] / proxy_total,
                torch.tensor(expected_share[name]),
                atol=1e-5,
                rtol=0,
            )

        budget = 12_389
        metadata.final_score = final
        keep = torch.topk(final, k=budget, largest=True, sorted=False).indices.sort().values
        key = torch.randn(1, 2, total, 8)
        value = torch.randn(1, 2, total, 8)
        pruned_key = key.index_select(2, keep)
        pruned_value = value.index_select(2, keep)
        pruned_metadata = gather_metadata(metadata, keep)

        self.assertEqual(pruned_key.shape[2], budget)
        self.assertEqual(pruned_value.shape[2], budget)
        self.assertEqual(pruned_metadata.final_score.numel(), budget)
        self.assertEqual(
            int(pruned_metadata.is_special.sum()), int(metadata.is_special.sum())
        )


if __name__ == "__main__":
    unittest.main()

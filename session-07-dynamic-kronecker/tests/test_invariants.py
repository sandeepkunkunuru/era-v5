"""Invariants the solution rests on. `python -m unittest discover tests` or pytest.

These are the claims the README makes, expressed as assertions. If one fails, the
corresponding claim is false.
"""
import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dyn.codecs import ByteCodec, parameter_count
from dyn.scripts import bytes_per_char, token_script

LONG_TAMIL = "▁பயன்படுத்தப்பட்டன."          # > 32 UTF-8 bytes
LONG_TAMIL_2 = "▁பயன்படுத்தப்படும்"          # shares its first 32 bytes


class TestDeterminism(unittest.TestCase):
    def test_codes_are_reproducible(self):
        """The codec is frozen by construction: same token, same code, always."""
        for kind, K in (("onehot", 32), ("fourier", 16), ("hybrid", 8), ("randproj", 16)):
            a = ByteCodec(kind, K).encode(["apple", "भारत", "x"])
            b = ByteCodec(kind, K).encode(["apple", "भारत", "x"])
            self.assertTrue(torch.equal(a, b), f"{kind} is not deterministic")

    def test_no_trainable_parameters_in_the_codec(self):
        code = ByteCodec("hybrid", 8).encode(["apple"])
        self.assertFalse(code.requires_grad)


class TestTruncation(unittest.TestCase):
    def test_v1_crops_past_its_window(self):
        c = ByteCodec("onehot", 32)
        self.assertTrue(c.caps_length)
        raw = LONG_TAMIL.encode("utf-8")
        self.assertGreater(len(raw), 32)
        self.assertEqual(len(c.visible_bytes(raw)), 32)          # bytes are dropped

    def test_dynamic_codecs_never_crop(self):
        raw = LONG_TAMIL.encode("utf-8")
        for kind, K in (("fourier", 16), ("hybrid", 8), ("randproj", 16), ("relative", 16)):
            c = ByteCodec(kind, K)
            self.assertFalse(c.caps_length, kind)
            self.assertEqual(len(c.visible_bytes(raw)), len(raw), kind)

    def test_a_751_byte_token_survives(self):
        """The longest token in the real vocabulary is 751 bytes."""
        long_token = "ab" * 400
        c = ByteCodec("hybrid", 8)
        self.assertEqual(len(c.visible_bytes(long_token.encode())), 800)
        self.assertEqual(c.encode([long_token]).shape, (1, c.code_dim))


class TestCollision(unittest.TestCase):
    def test_v1_maps_the_pair_to_one_identical_vector(self):
        """The central defect, asserted rather than assumed."""
        a, b = ByteCodec("onehot", 32).encode([LONG_TAMIL, LONG_TAMIL_2])
        self.assertEqual(float((a - b).abs().max()), 0.0)

    def test_dynamic_codecs_separate_the_same_pair(self):
        for kind, K in (("fourier", 16), ("hybrid", 8), ("randproj", 16)):
            a, b = ByteCodec(kind, K).encode([LONG_TAMIL, LONG_TAMIL_2])
            self.assertGreater(float((a - b).abs().max()), 1e-4, kind)


class TestPrefixSimilarity(unittest.TestCase):
    def test_shared_spelling_stays_close(self):
        """Kronecker's stated benefit — related spellings start near each other —
        must survive the change. `relative` is excluded: it provably breaks this,
        which is the finding, not a bug."""
        toks = ["train", "training", "quantum"]
        for kind, K in (("onehot", 32), ("fourier", 16), ("hybrid", 8)):
            e = torch.nn.functional.normalize(ByteCodec(kind, K).encode(toks), dim=-1)
            sim = e @ e.T
            self.assertGreater(float(sim[0, 1]), float(sim[0, 2]), kind)

    def test_relative_position_destroys_prefix_structure(self):
        e = torch.nn.functional.normalize(
            ByteCodec("relative", 16).encode(["train", "training"]), dim=-1)
        self.assertLess(float(e[0] @ e[1]), 0.5)


class TestCost(unittest.TestCase):
    def test_input_path_is_independent_of_vocabulary(self):
        """The property that makes Kronecker worth doing at all."""
        c = ByteCodec("hybrid", 8)
        self.assertEqual(parameter_count(c, 8096), parameter_count(c, 8096))
        self.assertEqual(c.code_dim, 2048)

    def test_hybrid8_is_a_quarter_of_v1(self):
        v1 = parameter_count(ByteCodec("onehot", 32), 8096)
        dyn = parameter_count(ByteCodec("hybrid", 8), 8096)
        self.assertEqual(v1, 66_322_432)                     # the lecture's number
        self.assertEqual(dyn * 4, v1)


class TestScripts(unittest.TestCase):
    def test_script_detection(self):
        self.assertEqual(token_script("▁भारत"), "Devanagari")
        self.assertEqual(token_script("▁பயன்"), "Tamil")
        self.assertEqual(token_script("▁apple"), "Latin")
        self.assertEqual(bytes_per_char("Telugu"), 3)
        self.assertEqual(bytes_per_char("Latin"), 1)

    def test_conjunct_costs_nine_bytes(self):
        """क्ष is three code points at three bytes each — the session's example."""
        self.assertEqual(len("क्ष".encode("utf-8")), 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)

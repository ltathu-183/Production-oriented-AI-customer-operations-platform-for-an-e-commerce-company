import unittest

import torch

from models import MiniUNet, SimpleSegNet
from utils import confusion_counts, metrics_from_counts


class MetricsTests(unittest.TestCase):
    def test_metrics_match_known_confusion_matrix(self):
        metrics = metrics_from_counts(tp=6, fp=2, fn=3, tn=9)

        self.assertAlmostEqual(metrics["precision"], 0.75)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 12 / 17)
        self.assertAlmostEqual(metrics["dice"], 12 / 17)
        self.assertAlmostEqual(metrics["iou"], 6 / 11)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)

    def test_zero_denominators_are_finite(self):
        metrics = metrics_from_counts(tp=0, fp=0, fn=0, tn=0)
        self.assertEqual(set(metrics.values()), {0.0})

    def test_confusion_counts_uses_pet_as_positive_class(self):
        predicted_classes = torch.tensor([[[0, 1], [1, 0]]])
        logits = torch.nn.functional.one_hot(predicted_classes, num_classes=2)
        logits = logits.permute(0, 3, 1, 2).float()
        target = torch.tensor([[[0, 1], [0, 1]]])

        self.assertEqual(confusion_counts(logits, target), (1, 1, 1, 1))


class ModelShapeTests(unittest.TestCase):
    def test_scratch_models_preserve_spatial_size(self):
        x = torch.randn(1, 3, 64, 64)
        for model in (SimpleSegNet(), MiniUNet()):
            with self.subTest(model=type(model).__name__):
                with torch.no_grad():
                    output = model(x)
                self.assertEqual(output.shape, (1, 2, 64, 64))


if __name__ == "__main__":
    unittest.main()

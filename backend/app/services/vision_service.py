import hashlib
from pathlib import Path
from uuid import UUID

import cv2


class VisionService:
    def analyze(self, image_path: str, case_id: UUID, image_id: UUID) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            width, height = 1024, 768
        else:
            height, width = img.shape[:2]

        digest = hashlib.sha256(Path(image_path).read_bytes()).digest()
        base_x = 0.18 + (digest[0] / 255.0) * 0.18
        base_y = 0.28 + (digest[1] / 255.0) * 0.22

        findings = [
            self._finding("caries", 0.71, width, height, base_x, base_y, "19"),
            self._finding("bone_loss", 0.63, width, height, min(base_x + 0.32, 0.72), min(base_y + 0.18, 0.68), "30"),
        ]
        return {
            "model_name": "mock-dental-vision",
            "model_version": "demo-0.1",
            "case_id": str(case_id),
            "image_id": str(image_id),
            "image_width": width,
            "image_height": height,
            "findings": findings,
        }

    @staticmethod
    def _finding(category: str, confidence: float, width: int, height: int, x_ratio: float, y_ratio: float, tooth: str) -> dict:
        x1 = int(width * x_ratio)
        y1 = int(height * y_ratio)
        x2 = min(width - 1, x1 + int(width * 0.13))
        y2 = min(height - 1, y1 + int(height * 0.11))
        polygon = [[x1, y1], [x2, y1 + 8], [x2 - 10, y2], [x1 + 6, y2 - 4]]
        return {
            "category": category,
            "confidence": confidence,
            "tooth_number": tooth,
            "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
            "polygon": polygon,
        }


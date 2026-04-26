"""
Face embedding using OpenCV SFace (FaceRecognizerSF).

SFace produces a 128-dimensional face embedding vector from an aligned face image.
It uses alignCrop() with the 5 landmarks from YuNet to normalize face pose
before feature extraction, significantly improving recognition accuracy.

Model: face_recognition_sface_2021dec.onnx (~37 MB)
Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
"""
import cv2
import numpy as np
from typing import Optional

from .config import SFACE_MODEL


class FaceEmbedder:
    """
    Face embedder using OpenCV SFace.

    Uses SFace's built-in alignment (alignCrop) with YuNet landmarks
    to produce normalized 128-dim face embeddings.
    """

    def __init__(self):
        model_path = str(SFACE_MODEL)
        if not SFACE_MODEL.exists():
            raise FileNotFoundError(
                f"SFace model not found at {model_path}. "
                "Run: python download_models.py"
            )

        self.recognizer = cv2.FaceRecognizerSF.create(
            model=model_path,
            config="",
        )

    def get_embedding(self, frame: np.ndarray, face_detection: np.ndarray) -> np.ndarray:
        """
        Extract a 128-dim face embedding from the frame using face detection info.

        Args:
            frame: Original full frame (BGR).
            face_detection: Raw detection row from YuNet containing bbox + 5 landmarks.
                           Shape: (15,) = [x,y,w,h, x_re,y_re, x_le,y_le, x_nt,y_nt, x_rcm,y_rcm, x_lcm,y_lcm, score]

        Returns:
            numpy array of shape (1, 128) – L2-normalized embedding.
        """
        # Align and crop face using landmarks from YuNet
        aligned_face = self.recognizer.alignCrop(frame, face_detection)
        # Extract 128-dim embedding
        embedding = self.recognizer.feature(aligned_face)
        return embedding  # shape: (1, 128), already L2-normalized

    def get_embedding_from_crop(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Fallback: Extract embedding from a pre-cropped face image (no alignment).
        Less accurate than get_embedding() but works without landmarks.

        Args:
            face_crop: Cropped face image (BGR), any size (will be resized internally).

        Returns:
            numpy array of shape (1, 128) – L2-normalized embedding.
        """
        # Resize to SFace expected input (112x112)
        resized = cv2.resize(face_crop, (112, 112))
        embedding = self.recognizer.feature(resized)
        return embedding

    def match(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings using SFace's built-in matcher.

        Returns:
            Cosine similarity score (higher = more similar).
            SFace recommended threshold: 0.363
        """
        return self.recognizer.match(
            emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE
        )
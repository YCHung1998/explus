import cv2
import numpy as np


class TextureEngine:
    """Engine for computing Local Binary Patterns (LBP) and Local Binary Similarity Patterns (LBSP).

    This engine supports texture-based change detection by comparing local pixel
    neighborhoods and calculating Hamming distances between descriptors.
    """

    def __init__(
        self,
        texture_sensitivity_ratio: float = 0.15,
        texture_diff_threshold: int = 5,
        dark_region_noise_floor: int = 50,
    ):
        """Initializes the TextureEngine with detection parameters.

        Args:
            texture_sensitivity_ratio: Relative intensity factor used for dynamic thresholding in LBSP.
            texture_diff_threshold: Threshold for the number of different bits to consider a change.
            dark_region_noise_floor: Minimum threshold to prevent noise in dark regions.
        """
        self.texture_sensitivity_ratio = texture_sensitivity_ratio
        self.dark_region_noise_floor = dark_region_noise_floor
        self.texture_diff_threshold = texture_diff_threshold

        # LBSP 5x5 Neighborhood definition (16 sampling points)
        # Using a 16-bit descriptor allows for rich texture representation.
        # self.offsets = [
        #     (-2, -2),           (-2,  0),           (-2,  2),
        #             (-1, -1), (-1,  0), (-1,  1),
        #     ( 0, -2), ( 0, -1),           ( 0,  1), ( 0,  2),
        #             ( 1, -1), ( 1,  0), ( 1,  1),
        #     ( 2, -2),           ( 2,  0),           ( 2,  2),
        # ]
        self.offsets = [
            (-2, -2),           (-2,  0),           (-2,  2),
                    (-1, -1), (-1,  0), (-1,  1),
            ( 0, -2), ( 0, -1),           ( 0,  1), ( 0,  2),
                    ( 1, -1), ( 1,  0), ( 1,  1),
            ( 2, -2),           ( 2,  0),           ( 2,  2),
        ]

    def compute_lbp(
        self,
        gray_img: np.ndarray,
        ref_img: np.ndarray = None,
    ) -> np.ndarray:
        """Computes the standard Local Binary Pattern (LBP) for an image.

        Args:
            gray_img: Input grayscale image (H, W).
            ref_img: Optional reference image. If provided, LBP is computed relative
                     to this image's intensities.

        Returns:
            A 16-bit unsigned integer array representing the LBP texture map.
        """
        h, w = gray_img.shape
        lbp = np.zeros((h, w), dtype=np.uint16)

        # Use float32 for centers to prevent overflow during comparisons.
        center = (
            ref_img.astype(np.float32)
            if ref_img is not None
            else gray_img.astype(np.float32)
        )

        # Pad image to handle boundaries (2-pixel border for 5x5 kernel).
        padded = cv2.copyMakeBorder(gray_img, 2, 2, 2, 2, cv2.BORDER_REFLECT)

        for i, (dy, dx) in enumerate(self.offsets):
            neighbor = padded[2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w].astype(
                np.float32
            )
            # Standard LBP logic: bit is 1 if neighbor is brighter or equal to center.
            mask = (neighbor >= center).astype(np.uint16)
            lbp |= mask << i

        return lbp

    def compute_lbsp(
        self,
        gray_img: np.ndarray,
        ref_img: np.ndarray = None,
    ) -> tuple:
        """Computes the Local Binary Similarity Pattern (LBSP).

        LBSP is more robust to illumination changes than standard LBP as it uses
        a dynamic threshold based on local intensity.

        Args:
            gray_img: Input grayscale image (H, W).
            ref_img: Reference image for cross-comparison.

        Returns:
            tuple: (lbsp_map, last_diff_map)
                   lbsp_map: 16-bit texture descriptor.
                   last_diff_map: Raw absolute difference of the final neighbor processed.
        """
        h, w = gray_img.shape
        lbsp = np.zeros((h, w), dtype=np.uint16)

        center = (
            ref_img.astype(np.float32)
            if ref_img is not None
            else gray_img.astype(np.float32)
        )

        # Calculate dynamic threshold: T(x) = max(noise_floor, rel_threshold * I(x))
        # This prevents flickering in dark areas.
        dynamic_thresh = np.clip(
            self.texture_sensitivity_ratio * center + 1.0,
            self.dark_region_noise_floor,
            255,
        ).astype(np.float32)

        padded = cv2.copyMakeBorder(gray_img, 2, 2, 2, 2, cv2.BORDER_REFLECT)
        diff = None

        for i, (dy, dx) in enumerate(self.offsets):
            neighbor = padded[2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w].astype(
                np.float32
            )
            diff = np.abs(neighbor - center)
            # LBSP similarity logic: bit is 1 if neighbor is "similar" to center.
            mask = (diff <= dynamic_thresh).astype(np.uint16)
            lbsp |= mask << i

        return lbsp, diff

    def get_hamming_map(
        self,
        lbsp1: np.ndarray,
        lbsp2: np.ndarray,
        texture_diff_threshold: int = None,
    ) -> tuple:
        """Calculates the Hamming distance between two texture maps.

        Args:
            lbsp1: First LBSP/LBP texture map.
            lbsp2: Second LBSP/LBP texture map.
            texture_diff_threshold: Distance threshold. Defaults to self.texture_diff_threshold.

        Returns:
            tuple: (binary_change_mask, distance_map)
                   binary_change_mask: Boolean mask where True indicates texture change.
                   distance_map: Integer map of Hamming distances.
        """
        if texture_diff_threshold is None:
            texture_diff_threshold = self.texture_diff_threshold

        # XOR identifies bits that are different.
        xor_res = np.bitwise_xor(lbsp1, lbsp2)

        # Efficient bit counting for 16-bit integers.
        # We use a vectorized approach by summing shifts (faster than bin().count()).
        dist = np.zeros(lbsp1.shape, dtype=np.uint8)
        temp_xor = xor_res.copy()
        for _ in range(16):
            dist += (temp_xor & 1).astype(np.uint8)
            temp_xor >>= 1

        change_mask = dist > texture_diff_threshold
        return change_mask, dist


# Example Usage:
if __name__ == "__main__":
    _engine = TextureEngine(texture_sensitivity_ratio=0.4, texture_diff_threshold=4)
    for _ in range(10):
        size = np.array([480, 640])
        img1 = np.random.randint(0, 255, tuple(size), dtype=np.uint8)
        img2 = np.random.randint(0, 255, tuple(size), dtype=np.uint8)
        for engine in [_engine]:
            # Dummy images
            lbsp_map, _ = engine.compute_lbsp(img1)
            lbsp_map_ref, _ = engine.compute_lbsp(img2)

            changes, dist_map = engine.get_hamming_map(lbsp_map, lbsp_map_ref)
            print(f"Mean Hamming Distance: {np.mean(dist_map):.2f}")

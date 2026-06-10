import numpy as np
from typing import (
    Tuple,
    Dict,
    Any,
    Union,
    Optional,
)

import math
import cv2
from scipy import optimize

import json
import pandas as pd
import matplotlib.pyplot as plt


class LetterBox:
    """
    Resize and pad image while preserving aspect ratio (LetterBox technique).

    This class handles the forward transformation of images and their labels
    (bounding boxes), and the inverse transformation of model outputs (BBoxes, masks).
    It also provides utility methods to transform bounding boxes based on shape only.
    """

    def __init__(
        self,
        new_shape: Tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        """
        Initialize LetterBox object. (Configuration is saved here)
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center
        self.padding_value = padding_value
        self.interpolation = interpolation
        self.params: Optional[Dict[str, Any]] = None

    @staticmethod
    def get_transform_params(
        original_shape: Tuple[int, int],
        new_shape: Tuple[int, int],
        auto: bool,
        scale_fill: bool,
        scaleup: bool,
        center: bool,
        stride: int,
    ) -> Dict[str, Any]:
        """
        [Static/Independent Method] Calculates the transformation parameters based on shapes.

        Args:
            original_shape (Tuple[int, int]): Original image shape (H, W).
            new_shape (Tuple[int, int]): Target image shape (H, W).
            ... (other LetterBox parameters)

        Returns:
            Dict[str, Any]: Transformation parameters.
        """
        shape = original_shape
        if isinstance(new_shape, int):
            new_shape_tuple = (new_shape, new_shape)
        else:
            new_shape_tuple = new_shape

        # Scale ratio (new / old)
        r = min(new_shape_tuple[0] / shape[0], new_shape_tuple[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        # Calculate unpadded new size (W, H)
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        # Calculate total padding (W, H)
        dw, dh = (
            new_shape_tuple[1] - new_unpad[0],
            new_shape_tuple[0] - new_unpad[1],
        )

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scale_fill:
            dw, dh = 0.0, 0.0

        if center:
            dw /= 2
            dh /= 2

        # Calculate actual top and left padding pixels
        top = int(round(dh - 0.1)) if center else 0
        left = int(round(dw - 0.1)) if center else 0

        # Store parameters
        params = {
            "scale_ratio": r,
            "padding_left": left,
            "padding_top": top,
            "original_shape": shape,
            "new_unpad": new_unpad,
            "is_scale_fill": scale_fill,
        }
        return params

    def _calculate_params(self, shape: Tuple[int, int]) -> Dict[str, Any]:
        """Calculates the transformation parameters (instance wrapper)."""
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        return self.get_transform_params(
            original_shape=shape,
            new_shape=new_shape,
            auto=self.auto,
            scale_fill=self.scale_fill,
            scaleup=self.scaleup,
            center=self.center,
            stride=self.stride,
        )

    @staticmethod
    def forward_bbox_transform(
        labels: np.ndarray, params: Dict[str, Any], new_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        [Static/Independent Method] Transforms bounding boxes using pre-calculated parameters.
        Assumes labels[:, :4] contains [x1, y1, x2, y2] coordinates (un-normalized pixels).
        """
        if labels.size == 0:
            return labels

        labels = labels.copy().astype(np.float32)

        r = params["scale_ratio"]
        padding_left = params["padding_left"]
        padding_top = params["padding_top"]
        is_scale_fill = params["is_scale_fill"]

        # 1. Apply Forward Transformation
        if is_scale_fill:
            # ScaleFill (Stretch)
            target_w, target_h = new_shape[1], new_shape[0]
            original_h, original_w = params["original_shape"]

            r_w = target_w / original_w
            r_h = target_h / original_h

            labels[:, [0, 2]] *= r_w
            labels[:, [1, 3]] *= r_h

        else:
            # LetterBox (Standard Mode)
            # a. Forward Scaling
            labels[:, :4] *= r
            # b. Forward Translation (Add Padding)
            labels[:, [0, 2]] += padding_left
            labels[:, [1, 3]] += padding_top

        # 2. Clip bounding boxes
        new_w, new_h = new_shape[1], new_shape[0]
        labels[:, [0, 2]] = np.clip(labels[:, [0, 2]], 0, new_w)
        labels[:, [1, 3]] = np.clip(labels[:, [1, 3]], 0, new_h)

        # Ensure x1 <= x2 and y1 <= y2
        labels[:, [0, 2]] = np.sort(labels[:, [0, 2]], axis=1)
        labels[:, [1, 3]] = np.sort(labels[:, [1, 3]], axis=1)

        return labels

    def _forward_transform_labels(self, labels: np.ndarray) -> np.ndarray:
        """Transforms labels using instance state (internal wrapper)."""
        if self.params is None:
            raise ValueError(
                "Parameters not calculated. Call __call__(image) first."
            )

        return self.forward_bbox_transform(
            labels=labels, params=self.params, new_shape=self.new_shape
        )

    def transform_box_from_shape(
        self, labels: np.ndarray, original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        [Public Utility] Transforms bounding boxes using only the shape information,
        applying the instance's LetterBox configuration.

        Args:
            labels (np.ndarray): Labels array (N, 4+) where the first 4 columns are [x1, y1, x2, y2].
            original_shape (Tuple[int, int]): The original image shape (H, W).

        Returns:
            np.ndarray: The transformed labels in the new shape space.
        """
        # 1. Calculate parameters based on the provided original_shape
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape_tuple = (new_shape, new_shape)
        else:
            new_shape_tuple = new_shape

        params = self.get_transform_params(
            original_shape=original_shape,
            new_shape=new_shape_tuple,
            auto=self.auto,
            scale_fill=self.scale_fill,
            scaleup=self.scaleup,
            center=self.center,
            stride=self.stride,
        )

        # 2. Apply the transformation using the calculated parameters
        transformed_labels = self.forward_bbox_transform(
            labels=labels, params=params, new_shape=new_shape_tuple
        )

        return transformed_labels

    def __call__(
        self, image: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        # ... (Image transformation logic remains the same, but relies on
        # self._calculate_params and self._forward_transform_labels)

        img = image
        shape = img.shape[:2]  # Current shape [height, width]

        # 1. Calculate and store transformation parameters
        self.params = self._calculate_params(shape)

        # Get calculated results for image transformation
        r = self.params["scale_ratio"]
        new_unpad = self.params["new_unpad"]  # W, H
        left = self.params["padding_left"]
        top = self.params["padding_top"]

        # Recalculate full padding needed for cv2.copyMakeBorder
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:
            dw, dh = 0.0, 0.0

        bottom = int(round(dh - top))
        right = int(round(dw - left))

        # 2. Perform image resizing
        if shape[::-1] != new_unpad and not self.scale_fill:
            img = cv2.resize(img, new_unpad, interpolation=self.interpolation)
            if img.ndim == 2:
                img = img[..., None]

        # 3. Perform image padding
        h, w, c = img.shape
        if not self.scale_fill:
            # LetterBox Padding
            if c == 3:
                img = cv2.copyMakeBorder(
                    img,
                    top,
                    bottom,
                    left,
                    right,
                    cv2.BORDER_CONSTANT,
                    value=(self.padding_value,) * 3,
                )
            else:  # multispectral
                pad_img = np.full(
                    (h + top + bottom, w + left + right, c),
                    fill_value=self.padding_value,
                    dtype=img.dtype,
                )
                pad_img[top : top + h, left : left + w] = img
                img = pad_img
        else:
            # ScaleFill (Stretch) - Resize directly to target size
            img = cv2.resize(
                img,
                (new_shape[1], new_shape[0]),
                interpolation=self.interpolation,
            )

        # 4. Perform label transformation
        if labels is not None:
            transformed_labels = self._forward_transform_labels(labels)
            return img, transformed_labels

        return img

    def unletterbox_bbox(
        self, bbox: np.ndarray, astype: str = "float"
    ) -> np.ndarray:
        # ... (Unchanged logic, relies on self.params being set by __call__)
        """
        Transforms bounding boxes from the padded/resized space back to the original image space.
        """
        if self.params is None:
            raise ValueError(
                "Must call the LetterBox instance (i.e., letterbox(image)) first to calculate parameters."
            )

        if bbox.size == 0:
            return (
                bbox.astype(np.float32)
                if astype == "float"
                else bbox.astype(np.int32)
            )

        bbox = bbox.copy().astype(np.float32)

        params = self.params
        r = params["scale_ratio"]
        padding_left = params["padding_left"]
        padding_top = params["padding_top"]
        original_h, original_w = params["original_shape"]
        is_scale_fill = params["is_scale_fill"]

        # 1. Apply Inverse Transformation
        if is_scale_fill:
            target_w, target_h = self.new_shape[1], self.new_shape[0]
            r_w = original_w / target_w
            r_h = original_h / target_h
            bbox[:, [0, 2]] *= r_w
            bbox[:, [1, 3]] *= r_h
        else:
            # a. Inverse Translation (Remove Padding)
            bbox[:, [0, 2]] -= padding_left
            bbox[:, [1, 3]] -= padding_top
            # b. Inverse Scaling
            bbox[:, :4] /= r

        # 2. Clip bounding boxes
        bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], 0, original_w)
        bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], 0, original_h)

        # 3. Ensure x1 <= x2 and y1 <= y2
        bbox[:, [0, 2]] = np.sort(bbox[:, [0, 2]], axis=1)
        bbox[:, [1, 3]] = np.sort(bbox[:, [1, 3]], axis=1)

        # 4. Convert data type
        if astype == "int":
            return bbox.astype(np.int32)
        elif astype == "float":
            return bbox
        else:
            raise ValueError(
                f"Invalid astype '{astype}'. Must be 'float' or 'int'."
            )

    def unletterbox_mask(self, mask: np.ndarray) -> np.ndarray:
        # ... (Unchanged logic, relies on self.params being set by __call__)
        """
        Transforms an instance segmentation mask from the padded/resized space back to the original image space.
        """
        if self.params is None:
            raise ValueError(
                "Must call the LetterBox instance (i.e., letterbox(image)) first to calculate parameters."
            )

        params = self.params
        padding_left = params["padding_left"]
        padding_top = params["padding_top"]
        new_unpad_w, new_unpad_h = params["new_unpad"]  # (W, H)
        original_h, original_w = params["original_shape"]  # (H, W)
        is_scale_fill = params["is_scale_fill"]

        # 1. Remove Padding / Crop
        if not is_scale_fill:
            mask = mask[
                padding_top : padding_top + new_unpad_h,
                padding_left : padding_left + new_unpad_w,
            ]

        # 2. Inverse Scaling
        mask = cv2.resize(
            mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR
        )

        return mask


class LetterBox_new:
    """Resize and pad image while preserving aspect ratio (LetterBox technique).
    
    Supports Hybrid Padding: A mix of Mirror padding (inner) and Constant padding (outer).
    """

    def __init__(
        self,
        new_shape: Tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
        # [NEW FEATURE] Controls the ratio of mirror padding (0.0 to 1.0)
        mirror_ratio: float = 0.0, 
    ):
        """Initialize LetterBox object.

        Args:
            new_shape (Tuple[int, int]): Target size (height, width).
            auto (bool): Minimum rectangle padding (stride aligned).
            scale_fill (bool): Stretch to fit (no padding).
            scaleup (bool): Allow scaling up.
            center (bool): Center the image.
            stride (int): Stride alignment.
            padding_value (int): Color value for constant padding.
            interpolation (int): OpenCV interpolation method.
            border_mode (int): Border mode for the *mirror* part. 
                               Use cv2.BORDER_REFLECT_101 or cv2.BORDER_REFLECT.
                               If cv2.BORDER_CONSTANT is used here, mirror_ratio is ignored.
            mirror_ratio (float): Ratio of padding to apply as mirror/reflection.
                                  0.0 = All Constant, 1.0 = All Mirror.
                                  0.5 = Half Mirror (inner), Half Constant (outer).
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center
        self.padding_value = padding_value
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.mirror_ratio = np.clip(mirror_ratio, 0.0, 1.0) # Ensure valid range
        self.params: Optional[Dict[str, Any]] = None
    
    @staticmethod
    def get_transform_params(
        original_shape: Tuple[int, int],
        new_shape: Tuple[int, int],
        auto: bool,
        scale_fill: bool,
        scaleup: bool,
        center: bool,
        stride: int,
    ) -> Dict[str, Any]:
        shape = original_shape
        if isinstance(new_shape, int):
            new_shape_tuple = (new_shape, new_shape)
        else:
            new_shape_tuple = new_shape

        r = min(new_shape_tuple[0] / shape[0], new_shape_tuple[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (
            new_shape_tuple[1] - new_unpad[0],
            new_shape_tuple[0] - new_unpad[1],
        )

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scale_fill:
            dw, dh = 0.0, 0.0

        if center:
            dw /= 2
            dh /= 2

        top = int(round(dh - 0.1)) if center else 0
        left = int(round(dw - 0.1)) if center else 0

        params = {
            "scale_ratio": r,
            "padding_left": left,
            "padding_top": top,
            "original_shape": shape,
            "new_unpad": new_unpad,
            "is_scale_fill": scale_fill,
        }
        return params

    def _calculate_params(self, shape: Tuple[int, int]) -> Dict[str, Any]:
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        return self.get_transform_params(
            original_shape=shape,
            new_shape=new_shape,
            auto=self.auto,
            scale_fill=self.scale_fill,
            scaleup=self.scaleup,
            center=self.center,
            stride=self.stride,
        )

    @staticmethod
    def forward_bbox_transform(
        labels: np.ndarray, params: Dict[str, Any], new_shape: Tuple[int, int]
    ) -> np.ndarray:
        if labels.size == 0:
            return labels
        labels = labels.copy().astype(np.float32)
        r = params["scale_ratio"]
        padding_left = params["padding_left"]
        padding_top = params["padding_top"]
        is_scale_fill = params["is_scale_fill"]

        if is_scale_fill:
            target_w, target_h = new_shape[1], new_shape[0]
            original_h, original_w = params["original_shape"]
            r_w = target_w / original_w
            r_h = target_h / original_h
            labels[:, [0, 2]] *= r_w
            labels[:, [1, 3]] *= r_h
        else:
            labels[:, :4] *= r
            labels[:, [0, 2]] += padding_left
            labels[:, [1, 3]] += padding_top

        new_w, new_h = new_shape[1], new_shape[0]
        labels[:, [0, 2]] = np.clip(labels[:, [0, 2]], 0, new_w)
        labels[:, [1, 3]] = np.clip(labels[:, [1, 3]], 0, new_h)
        labels[:, [0, 2]] = np.sort(labels[:, [0, 2]], axis=1)
        labels[:, [1, 3]] = np.sort(labels[:, [1, 3]], axis=1)
        return labels
    
    def _forward_transform_labels(self, labels: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise ValueError("Parameters not calculated. Call __call__(image) first.")
        return self.forward_bbox_transform(labels=labels, params=self.params, new_shape=self.new_shape)

    # ... [transform_box_from_shape is unchanged] ...
    def transform_box_from_shape(self, labels: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape_tuple = (new_shape, new_shape)
        else:
            new_shape_tuple = new_shape
        params = self.get_transform_params(
            original_shape=original_shape,
            new_shape=new_shape_tuple,
            auto=self.auto,
            scale_fill=self.scale_fill,
            scaleup=self.scaleup,
            center=self.center,
            stride=self.stride,
        )
        return self.forward_bbox_transform(labels=labels, params=params, new_shape=new_shape_tuple)

    def __call__(
        self, image: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        img = image
        shape = img.shape[:2]  # [height, width]

        # 1. Calculate and store transformation parameters
        self.params = self._calculate_params(shape)

        new_unpad = self.params["new_unpad"]  # W, H
        left = self.params["padding_left"]
        top = self.params["padding_top"]

        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Calculate total required padding
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:
            dw, dh = 0.0, 0.0

        total_bottom = int(round(dh - top))
        total_right = int(round(dw - left))
        total_top = top
        total_left = left

        # 2. Perform image resizing
        if shape[::-1] != new_unpad and not self.scale_fill:
            img = cv2.resize(img, new_unpad, interpolation=self.interpolation)
            if img.ndim == 2:
                img = img[..., None]

        # 3. Perform image padding (The Hybrid Logic)
        if not self.scale_fill:
            h, w, c = img.shape
            # Decide split between Mirror (Inner) and Constant (Outer)
            # Only do hybrid if mirror_ratio > 0 and mode is NOT constant
            use_hybrid = (self.mirror_ratio > 0.0) and (self.border_mode != cv2.BORDER_CONSTANT)
            
            if use_hybrid:
                # -- Phase 3a: Calculate Split --
                # Inner (Mirror) part
                m_top = int(total_top * self.mirror_ratio)
                m_bottom = int(total_bottom * self.mirror_ratio)
                m_left = int(total_left * self.mirror_ratio)
                m_right = int(total_right * self.mirror_ratio)

                # Outer (Constant) part
                c_top = total_top - m_top
                c_bottom = total_bottom - m_bottom
                c_left = total_left - m_left
                c_right = total_right - m_right

                # -- Phase 3b: Apply Inner Mirror Padding --
                if any([m_top, m_bottom, m_left, m_right]):
                    img = cv2.copyMakeBorder(
                        img, m_top, m_bottom, m_left, m_right, 
                        self.border_mode # e.g., cv2.BORDER_REFLECT_101
                    )
                
                # -- Phase 3c: Apply Outer Constant Padding --
                # Note: Logic copied from original to handle multi-channel/gray consistency
                if any([c_top, c_bottom, c_left, c_right]):
                    if c == 3:
                        img = cv2.copyMakeBorder(
                            img, c_top, c_bottom, c_left, c_right,
                            cv2.BORDER_CONSTANT,
                            value=(self.padding_value,) * 3,
                        )
                    else:
                        # Re-calculate shape after mirror padding
                        curr_h, curr_w = img.shape[:2]
                        pad_img = np.full(
                            (curr_h + c_top + c_bottom, curr_w + c_left + c_right, c),
                            fill_value=self.padding_value,
                            dtype=img.dtype,
                        )
                        pad_img[c_top : c_top + curr_h, c_left : c_left + curr_w] = img
                        img = pad_img

            else:
                # -- Standard Behavior (All Constant OR All Mirror) --
                # If border_mode is CONSTANT, mirror_ratio is ignored (effectively 0)
                # If border_mode is REFLECT but mirror_ratio is 0, it acts as Constant (if implemented strictly)
                # But here, if mirror_ratio=0, we fall back to standard logic.
                
                # Note: To be fully robust, if ratio=0, we should force BORDER_CONSTANT
                # regardless of self.border_mode, OR respect self.border_mode fully.
                # Assuming original behavior: respect self.border_mode fully if not splitting.
                
                # Check for explicit Constant mode logic
                if self.border_mode == cv2.BORDER_CONSTANT:
                    if c == 3:
                        img = cv2.copyMakeBorder(
                            img, total_top, total_bottom, total_left, total_right,
                            cv2.BORDER_CONSTANT, value=(self.padding_value,) * 3
                        )
                    else:
                        pad_img = np.full(
                            (h + total_top + total_bottom, w + total_left + total_right, c),
                            fill_value=self.padding_value,
                            dtype=img.dtype,
                        )
                        pad_img[total_top : total_top + h, total_left : total_left + w] = img
                        img = pad_img
                else:
                    # Pure Mirror (ratio assumed 1.0 or user set mode but ratio=0 -> full mirror)
                    img = cv2.copyMakeBorder(
                        img, total_top, total_bottom, total_left, total_right,
                        self.border_mode
                    )

        else:
            # ScaleFill (Stretch)
            img = cv2.resize(
                img,
                (new_shape[1], new_shape[0]),
                interpolation=self.interpolation,
            )

        # 4. Perform label transformation
        if labels is not None:
            transformed_labels = self._forward_transform_labels(labels)
            return img, transformed_labels

        return img

    def unletterbox_bbox(self, bbox: np.ndarray, astype: str = "float") -> np.ndarray:
         # (Logic identical to previous version, uses self.params)
        if self.params is None: raise ValueError("Call letterbox first.")
        if bbox.size == 0: return bbox
        bbox = bbox.copy().astype(np.float32)
        params = self.params
        r = params["scale_ratio"]
        padding_left = params["padding_left"]
        padding_top = params["padding_top"]
        original_h, original_w = params["original_shape"]
        is_scale_fill = params["is_scale_fill"]

        if is_scale_fill:
            target_w, target_h = self.new_shape[1], self.new_shape[0]
            bbox[:, [0, 2]] *= original_w / target_w
            bbox[:, [1, 3]] *= original_h / target_h
        else:
            bbox[:, [0, 2]] -= padding_left
            bbox[:, [1, 3]] -= padding_top
            bbox[:, :4] /= r

        bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], 0, original_w)
        bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], 0, original_h)
        bbox[:, [0, 2]] = np.sort(bbox[:, [0, 2]], axis=1)
        bbox[:, [1, 3]] = np.sort(bbox[:, [1, 3]], axis=1)
        
        if astype == "int": return bbox.astype(np.int32)
        return bbox

    def unletterbox_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.params is None: raise ValueError("Call letterbox first.")
        params = self.params
        padding_left = params["padding_left"]
        padding_top = params["padding_top"]
        new_unpad_w, new_unpad_h = params["new_unpad"]
        original_h, original_w = params["original_shape"]
        is_scale_fill = params["is_scale_fill"]

        if not is_scale_fill:
            mask = mask[padding_top : padding_top + new_unpad_h, padding_left : padding_left + new_unpad_w]
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        return mask


def calculate_iou_by_bbox_xywh(bbox_a, bbox_b):
    """Intersection over Union from bounding boxes

    Args:
        bbox_a/b: int/float tuple (x, y, w, h)
    Returns:
        floating value with formula =
        Intersect(bbox_a, bbox_b) / Union (bbox_a, bbox_b)
    """
    assert len(bbox_a) == 4, "len(bbox_a) %d != 4" % len(bbox_a)
    assert len(bbox_b) == 4, "len(bbox_b) %d != 4" % len(bbox_b)
    # get intersection bbox
    tl_x = max(bbox_a[0], bbox_b[0])
    tl_y = max(bbox_a[1], bbox_b[1])
    br_x = min(bbox_a[0] + bbox_a[2], bbox_b[0] + bbox_b[2])
    br_y = min(bbox_a[1] + bbox_a[3], bbox_b[1] + bbox_b[3])
    if (tl_x <= br_x) and (tl_y <= br_y):  # if has intersection
        intersect_sz = (br_x - tl_x) * (br_y - tl_y) * 1.0
        # Denominator
        denominator = (
            bbox_a[2] * bbox_a[3] + bbox_b[2] * bbox_b[3] - intersect_sz
        )
        return intersect_sz / denominator
    # if no intersection
    return 0


def calculate_iou_by_bbox_xyxy(bbox_a, bbox_b):
    """Intersection over Union from bounding boxes

    Args:
        bbox_a/b: int/float tuple (x, y, x, y)
    Returns:
        floating value with formula =
        Intersect(bbox_a, bbox_b) / Union (bbox_a, bbox_b)
    """
    assert len(bbox_a) == 4, "len(bbox_a) %d != 4" % len(bbox_a)
    assert len(bbox_b) == 4, "len(bbox_b) %d != 4" % len(bbox_b)
    # get intersection bbox

    tl_x = max(bbox_a[0], bbox_b[0])
    tl_y = max(bbox_a[1], bbox_b[1])
    br_x = min(bbox_a[2], bbox_b[2])
    br_y = min(bbox_a[3], bbox_b[3])
    if (tl_x <= br_x) and (tl_y <= br_y):  # if has intersection
        intersect_sz = (br_x - tl_x) * (br_y - tl_y) * 1.0
        # Denominator
        denominator = (
            (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
            + (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
            - intersect_sz
        )
        return intersect_sz / denominator
    # if no intersection
    return 0


def calculate_iou_matrix(list_inst_a, list_inst_b, type="bbox", mode="xyxy"):
    """Calculate IoU matrix between two list of instances

    Args:
        type: str; bbox or contours
    """
    iou_matrix = np.zeros(
        (len(list_inst_a), len(list_inst_b)), dtype=np.float32
    )
    for idx_a, inst_a in enumerate(list_inst_a):
        for idx_b, inst_b in enumerate(list_inst_b):
            if type == "bbox":
                if mode == "xywh":
                    iou_matrix[idx_a, idx_b] = calculate_iou_by_bbox_xywh(
                        inst_a, inst_b
                    )
                elif mode == "xyxy":
                    iou_matrix[idx_a, idx_b] = calculate_iou_by_bbox_xyxy(
                        inst_a, inst_b
                    )
            else:
                assert False, 'unsupported type. must be "bbox"'
    return iou_matrix


def match_gt_pd_instances_per_class_per_img(
    list_gt_insts, list_pd_insts, iou_type="bbox", iou_thres=0.5
):
    """match ground truth and prediction instances per class per image

    Args:
        list_gt_insts: [[x, y, w, h], ...]
        list_pd_insts: [[x, y, w, h], ...]
        iou_type: 'bbox'
        iou_thres: float, above set as matched, lower set as not_tight
    Returns:
        matched_indices: list of (gt_row_idx, pd_row_idx) pair
        not_tight_indices: list of (gt_row_idx, pd_row_idx) pair
        gt_orphan_indices: list of orphan (gt_row_idx)
        pd_orphan_indices: list of orphan (pd_row_idx)
    """

    # init
    matched_indices = []
    # split into matched_indices and not_tight_indices
    ious = {"matched": [], "not_tight": []}
    not_tight_indices = []  # iou
    gt_orphan_indices = []
    pd_orphan_indices = []

    # collect shapes
    list_gt_idxs = list(range(len(list_gt_insts)))
    list_pd_idxs = list(range(len(list_pd_insts)))

    # if either list is empty
    if len(list_gt_idxs) == 0:
        pd_orphan_indices = list_pd_idxs
        return (
            matched_indices,
            not_tight_indices,
            gt_orphan_indices,
            pd_orphan_indices,
            ious,
        )
    if len(list_pd_idxs) == 0:
        gt_orphan_indices = list_gt_idxs
        return (
            matched_indices,
            not_tight_indices,
            gt_orphan_indices,
            pd_orphan_indices,
            ious,
        )

    # calculate iou matrix and get match indices
    iou_matrix = calculate_iou_matrix(list_gt_insts, list_pd_insts, iou_type)
    idx_row, idx_col = optimize.linear_sum_assignment(-iou_matrix)
    # organize matched indices
    for idx_a, idx_b in zip(idx_row, idx_col):
        iou = iou_matrix[idx_a, idx_b]
        if iou > iou_thres:
            matched_indices.append([list_gt_idxs[idx_a], list_pd_idxs[idx_b]])
            ious["matched"].append(iou)
        elif iou > 0:
            not_tight_indices.append(
                [list_gt_idxs[idx_a], list_pd_idxs[idx_b]]
            )
            ious["not_tight"].append(iou)

    # organize orphan indices
    _gt_matched_indices = ()
    _pd_matched_indices = ()
    _gt_not_tight_indices = ()
    _pd_not_tight_indices = ()
    if matched_indices:
        _gt_matched_indices, _pd_matched_indices = zip(*matched_indices)
    if not_tight_indices:
        _gt_not_tight_indices, _pd_not_tight_indices = zip(*not_tight_indices)
    gt_orphan_indices = [
        x
        for x in list_gt_idxs
        if x not in _gt_matched_indices + _gt_not_tight_indices
    ]
    pd_orphan_indices = [
        x
        for x in list_pd_idxs
        if x not in _pd_matched_indices + _pd_not_tight_indices
    ]
    return (
        matched_indices,
        not_tight_indices,
        gt_orphan_indices,
        pd_orphan_indices,
        ious,
    )


# like iou 分母
def union_bbox(bbox1, bbox2):
    """
    bbox1, bbox2: a  [x, y, w, h]
    return: a [x, y, w, h]
    """
    assert len(bbox1) == len(bbox2)

    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1 + w1, x2 + w2)
    y_max = max(y1 + h1, y2 + h2)

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def crop_image_nchw(image, bbox):
    x, y, w, h = bbox

    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)

    # 防止超出邊界
    _, _, H, W = image.shape
    x1 = max(0, min(x1, W))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H))
    y2 = max(0, min(y2, H))

    # NCHW slicing
    crop = image[:, :, y1:y2, x1:x2]

    return crop


def calculate_tiou(box_a, box_b):
    s_a, e_a = box_a
    s_b, e_b = box_b

    # max of starts, min of ends
    inter_s = max(s_a, s_b)
    inter_e = min(e_a, e_b)
    inter_len = max(0, inter_e - inter_s)

    # Union = Len_A + Len_B - Intersection
    duration_a = e_a - s_a
    duration_b = e_b - s_b
    union_len = duration_a + duration_b - inter_len

    if union_len <= 0:
        return 0.0
    return inter_len / union_len


def calculate_tiou_matrix(list_seg_a, list_seg_b):
    rows = len(list_seg_a)
    cols = len(list_seg_b)
    iou_matrix = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            iou_matrix[r, c] = calculate_tiou(
                list_seg_a[r],
                list_seg_b[c],
            )
    return iou_matrix


def match_temporal_segments(
    list_gt_segments,
    list_pd_segments,
    tiou_thres=0.5,
):
    """
    在時間軸上匹配 GT 和 PD (1D Domain)

    Args:
        list_gt_segments: [[start, end], ...] (Ground Truth)
        list_pd_segments: [[start, end], ...] (Prediction)
        tiou_thres: float, 判定為 Matched 的門檻 (例如 0.5)

    Returns:
        matched_indices: List of [gt_idx, pd_idx] (成功配對)
        not_tight_indices: List of [gt_idx, pd_idx] (有重疊但 IoU 不夠高)
        gt_orphan_indices: List of gt_idx (漏報 / Undercount / Miss)
        pd_orphan_indices: List of pd_idx (誤報 / False Alarm)
        ious: dict 紀錄詳細 IoU 數值
    """

    # init
    matched_indices = []
    ious = {"matched": [], "not_tight": []}
    not_tight_indices = []
    gt_orphan_indices = []
    pd_orphan_indices = []

    # indices helpers
    list_gt_idxs = list(range(len(list_gt_segments)))
    list_pd_idxs = list(range(len(list_pd_segments)))

    # --- Edge Case: 若其中一方為空 ---
    if len(list_gt_idxs) == 0:
        pd_orphan_indices = list_pd_idxs
        return (
            matched_indices,
            not_tight_indices,
            gt_orphan_indices,
            pd_orphan_indices,
            ious,
        )

    if len(list_pd_idxs) == 0:
        gt_orphan_indices = list_gt_idxs
        return (
            matched_indices,
            not_tight_indices,
            gt_orphan_indices,
            pd_orphan_indices,
            ious,
        )

    # --- Core Logic ---

    # 1. 計算 tIoU Matrix
    iou_matrix = calculate_tiou_matrix(list_gt_segments, list_pd_segments)

    # 2. 匈牙利演算法 (Hungarian Algorithm) 找最佳二分匹配
    # linear_sum_assignment 尋找最小 cost，所以我們要把 IoU 取負號 (找最大 IoU)
    row_indices, col_indices = optimize.linear_sum_assignment(-iou_matrix)

    # 3. 根據閾值分類匹配結果
    for r, c in zip(row_indices, col_indices):
        iou = iou_matrix[r, c]

        if iou >= tiou_thres:
            # 完美匹配 (True Positive)
            matched_indices.append([list_gt_idxs[r], list_pd_idxs[c]])
            ious["matched"].append(iou)
        elif iou > 0:
            # 有重疊但不夠精準 (Localization Error)
            not_tight_indices.append([list_gt_idxs[r], list_pd_idxs[c]])
            ious["not_tight"].append(iou)
            # 注意：這裡雖然不算 Matched，但被匈牙利演算法吃掉了，
            # 所以這對 GT 和 PD 都不會進入 Orphan (除非你希望 Loose match 也算 Miss/FA)

    # 4. 整理孤兒 (Orphans)

    # 先收集所有已經被分配掉的 index (包含 matched 和 not_tight)
    _gt_assigned = set(r for r, c in matched_indices + not_tight_indices)
    _pd_assigned = set(c for r, c in matched_indices + not_tight_indices)

    # 剩下的就是孤兒
    gt_orphan_indices = [x for x in list_gt_idxs if x not in _gt_assigned]
    pd_orphan_indices = [x for x in list_pd_idxs if x not in _pd_assigned]

    return (
        matched_indices,
        not_tight_indices,
        gt_orphan_indices,
        pd_orphan_indices,
        ious,
    )


def analyze_segments_and_gaps(json_files):
    durations = []
    gaps = []
    if not json_files:
        print("找不到任何 JSON 檔案。")
        return

    boundary_files = {"duration": set(), "gap": set()}
    
    for file_path in json_files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                root = data.get("results", data)

                video_segments = []
                for video_id, contents in root.items():
                    if video_id == "version" or not isinstance(contents, list):
                        continue
                    for ann in contents:
                        if "segment" in ann:
                            start, end = ann["segment"]
                            dur = end - start
                            durations.append(dur)
                            video_segments.append([start, end])
                            
                            # Check Duration Boundary (0.5s under default)
                            if round(dur, 2) <= 0.5:
                                boundary_files["duration"].add(file_path)

                # 統計相鄰間隙 (Gap Analysis)
                if len(video_segments) > 1:
                    video_segments.sort(key=lambda x: x[0])
                    for i in range(len(video_segments) - 1):
                        prev_end = video_segments[i][1]
                        curr_start = video_segments[i + 1][0]
                        gap = max(0, curr_start - prev_end)
                        gaps.append(gap)
                        
                        # Check Gap Boundary (0.1s under default)
                        if 0 < round(gap, 2) <= 0.11:
                            boundary_files["gap"].add(file_path)

            except Exception as e:
                print(f"解析 {file_path} 出錯: {e}")

    if not durations:
        print("沒有找到任何區間數據。")
        return

    # 數據處理
    df_dur = pd.Series(durations)
    df_gap = pd.Series(gaps)

    # 1. 輸出數值結果
    print("=== 區間長度統計 (Durations) ===")
    print(
        f"最短: {df_dur.min():.3f}s, 最長: {df_dur.max():.3f}s, 平均: {df_dur.mean():.3f}s"
    )
    if boundary_files["duration"]:
        print(f"  ● 觸發最短長度 (<=0.5s) 的檔案數量: {len(boundary_files['duration'])}")
        for f_path in sorted(list(boundary_files["duration"])):
            print(f"    - {f_path}")

    print("\n=== 相鄰間隙統計 (Gaps between segments) ===")
    if not df_gap.empty:
        print(f"最窄間隙: {df_gap.min():.3f}s")
        print(f"最寬間隙: {df_gap.max():.3f}s")
        print(f"平均間隙: {df_gap.mean():.3f}s")
        print(f"中位數間隙: {df_gap.median():.3f}s")
        if boundary_files["gap"]:
            print(f"  ● 觸發最窄間隙 (<=0.1s) 的檔案數量: {len(boundary_files['gap'])}")
            for f_path in sorted(list(boundary_files["gap"])):
                print(f"    - {f_path}")
    else:
        print("無相鄰區間數據。")

    # 2. 繪圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左圖：區間長度分布
    ax1.hist(df_dur, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    ax1.set_title("Segment Durations Distribution")
    ax1.set_xlabel("Seconds")
    ax1.set_ylabel("Frequency")
    ax1.axvline(
        df_dur.min(),
        color="orange",
        linestyle="--",
        label=f"Min: {df_dur.min():.2f}s",
    )
    ax1.axvline(
        df_dur.median(),
        color="red",
        linestyle="--",
        label=f"Median: {df_dur.median():.2f}s",
    )
    ax1.legend()

    # 右圖：相鄰間隙分布
    if not df_gap.empty:
        # 針對間隙通常很短的特性，可以限制一下繪圖範圍看細節 (例如只看 2秒內的間隙)
        ax2.hist(
            df_gap[df_gap < 2.0],
            bins=50,
            color="salmon",
            edgecolor="black",
            alpha=0.7,
        )
        ax2.set_title("Adjacent Gaps Distribution (Gaps < 2s)")
        ax2.set_xlabel("Seconds (Gap)")
        ax2.set_ylabel("Frequency")
        ax2.axvline(
            df_gap.min(),
            color="red",
            linestyle="--",
            label=f"Min: {df_gap.min():.2f}s",
        )
        ax2.axvline(
            df_gap.median(),
            color="blue",
            linestyle="--",
            label=f"Median: {df_gap.median():.2f}s",
        )
        # # 畫出 0.3s 的參考線
        # ax2.axvline(
        #     0.3,
        #     color="green",
        #     linestyle="-",
        #     linewidth=2,
        #     label="Proposed Union (0.3s)",
        # )
        ax2.legend()

    plt.tight_layout()
    plt.show()


# 使用方式：更換為你的 .json 檔案存放目錄
# analyze_segment_durations('./your_json_folder')

if __name__ == "__main__":
    import path_utils

    all_anno_video_paths = path_utils.get_files_recursive(
        # "/Users/eason.hung/Documents/Projects/explus/output/phash_0112_bdry_v2/predictions/",
        "/Users/eason.hung/Documents/Projects/explus/output/Block_m0_Backbone_P4_yolo/predictions/",
        # '/Users/eason.hung/Documents/Projects/explus/output/Block_0106/predictions/',
        # NOTE(Eason): vidat 精標之後取的後綴名稱
        supported_extensions=(".json",),
    )
    all_anno_video_paths = [
        fn for fn in all_anno_video_paths if "merge_data.json" not in fn
    ]
    analyze_segments_and_gaps(all_anno_video_paths)

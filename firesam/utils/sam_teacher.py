from typing import Optional, Tuple
import numpy as np
import torch


class ResizeLongestSide:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def get_preprocess_shape(self, oldh: int, oldw: int) -> Tuple[int, int]:
        scale = self.target_length / max(oldh, oldw)
        newh = int(round(oldh * scale))
        neww = int(round(oldw * scale))
        return newh, neww

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        import cv2

        h, w = image.shape[:2]
        newh, neww = self.get_preprocess_shape(h, w)
        if (newh, neww) == (h, w):
            return image
        return cv2.resize(image, (neww, newh), interpolation=cv2.INTER_LINEAR)

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        oldh, oldw = original_size
        newh, neww = self.get_preprocess_shape(oldh, oldw)
        scale = np.array([neww / oldw, newh / oldh], dtype=np.float32)
        return coords * scale

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        boxes = boxes.copy()
        coords = boxes.reshape(-1, 2)
        coords = self.apply_coords(coords, original_size)
        return coords.reshape(-1, 4)


class EdgeSAMFireTeacher:
    # Wrapper around an EdgeSAM model for fire segmentation KD.
    def __init__(self, sam_model: torch.nn.Module, device: torch.device) -> None:
        self.model = sam_model
        self.device = device
        img_size = getattr(getattr(self.model, "image_encoder", None), "img_size", 1024)
        self.transform = ResizeLongestSide(img_size)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_single(
        self,
        image_np: np.ndarray,
        box: Optional[np.ndarray],
        pos_points: np.ndarray,
        neg_points: np.ndarray,
        multimask_output: bool = False,
    ) -> torch.Tensor:
        """Predict fire mask logits for a single image.

        Args:
            image_np: (H, W, 3) uint8 RGB.
            box: [x1, y1, x2, y2] in original image coords, or None.
            pos_points: (N_pos, 2) points (x, y).
            neg_points: (N_neg, 2) points (x, y).
        Returns:
            mask_logits: (1, 1, H, W) tensor on the same device as the teacher.
        """
        orig_h, orig_w = image_np.shape[:2]
        input_image = self.transform.apply_image(image_np)
        input_h, input_w = input_image.shape[:2]

        image_t = torch.as_tensor(input_image, device=self.device).permute(2, 0, 1).unsqueeze(0)
        image_t = self.model.preprocess(image_t)

        features = self.model.image_encoder(image_t)

        # Build prompts
        all_points = []
        all_labels = []

        if pos_points is not None and pos_points.shape[0] > 0:
            all_points.append(pos_points)
            all_labels.append(np.ones((pos_points.shape[0],), dtype=np.int64))
        if neg_points is not None and neg_points.shape[0] > 0:
            all_points.append(neg_points)
            all_labels.append(np.zeros((neg_points.shape[0],), dtype=np.int64))

        if len(all_points) > 0:
            pts = np.concatenate(all_points, axis=0)
            labs = np.concatenate(all_labels, axis=0)
            pts = self.transform.apply_coords(pts, (orig_h, orig_w))
            pts_t = torch.as_tensor(pts, device=self.device)[None, :, :]
            labs_t = torch.as_tensor(labs, device=self.device)[None, :]
            points_tuple = (pts_t, labs_t)
        else:
            points_tuple = None

        if box is not None:
            box_arr = np.array(box, dtype=np.float32)[None, :]
            box_arr = self.transform.apply_boxes(box_arr, (orig_h, orig_w))
            boxes_t = torch.as_tensor(box_arr, device=self.device)
        else:
            boxes_t = None

        sparse_emb, dense_emb = self.model.prompt_encoder(
            points=points_tuple,
            boxes=boxes_t,
            masks=None,
        )

        low_res_masks, iou_preds = self.model.mask_decoder(
            image_embeddings=features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            num_multimask_outputs=3 if multimask_output else 1,
        )

        masks = self.model.postprocess_masks(
            low_res_masks,
            (input_h, input_w),
            (orig_h, orig_w),
        )

        # Use first mask by default
        if masks.shape[1] > 1:
            masks = masks[:, :1, :, :]
        return masks

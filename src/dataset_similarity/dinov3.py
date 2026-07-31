import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel


class DINOv3Classifier(PreTrainedModel):  # type: ignore[misc]
    """
    Image classification model built on top of DINOv3ViTModel.

    Combines the CLS token and the mean of the patch tokens into a single
    representation, then passes it through a linear classifier head. This
    mirrors the evaluation head used in the original DINOv3 paper and is
    compatible with the HuggingFace ``Trainer`` API.
    """

    base_model_prefix = "dinov3"
    config_class = DINOv3ViTConfig

    def __init__(self, config: DINOv3ViTConfig):
        """
        Initialise the classifier.

        Args:
            config: A ``DINOv3ViTConfig`` instance. ``config.num_labels`` controls
                the output size of the classification head; when it is 0 or negative
                the head is replaced with an ``nn.Identity``.
        """
        super().__init__(config)
        self.dinov3 = DINOv3ViTModel(config)
        self.num_labels = config.num_labels
        self.num_register_tokens = self.dinov3.config.num_register_tokens
        self.classifier = (
            nn.Linear(config.hidden_size * 2, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> ImageClassifierOutput | tuple[torch.Tensor, ...]:
        """
        Run a forward pass through the DINOv3 backbone and classification head.

        The representation fed to the classifier is the concatenation of the CLS
        token and the mean-pooled patch tokens, giving a vector of size
        ``2 * hidden_size``.

        Args:
            pixel_values: Batch of pre-processed images with shape ``[N, C, H, W]``.
            labels: Integer class indices with shape ``[N]``. When provided, a
                cross-entropy loss is computed and included in the output.
            output_attentions: Whether to return attention weights from the backbone.
            output_hidden_states: Whether to return all hidden states from the backbone.
            return_dict: If ``False``, returns a plain tuple instead of an
                ``ImageClassifierOutput`` dataclass.

        Returns:
            An ``ImageClassifierOutput`` (or a flat tuple when ``return_dict=False``)
            containing the optional loss, logits, and optionally hidden states and
            attentions.
        """
        outputs = self.dinov3(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hs = outputs.last_hidden_state  # [N, 1 + num_register_tokens + num_patches, D]
        cls_token = hs[:, 0]  # [N, D]
        patch_tokens = hs[:, 1 + self.num_register_tokens :]  # [N, num_patches, D]
        patch_mean = patch_tokens.mean(dim=1)  # [N, D]
        logits = self.classifier(
            torch.cat([cls_token, patch_mean], dim=-1)
        )  # [N, num_classes]
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        if return_dict is False:
            output = (logits, *outputs[1:])
            return (loss, *output) if loss is not None else output
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel


class DINOv3Classifier(PreTrainedModel):  # type: ignore[misc]
    config_class = DINOv3ViTConfig

    def __init__(self, config: DINOv3ViTConfig):
        super().__init__(config)
        self.dinov3 = DINOv3ViTModel(config)
        self.num_labels = config.num_labels
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
            output = (logits,) + outputs[1:]
            return (loss, output) if loss is not None else output
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

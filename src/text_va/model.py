import torch
from transformers import XLMRobertaForSequenceClassification


class XLMRobertaVA(XLMRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.output_activation = torch.tanh

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.output_activation(outputs.logits)
        outputs.logits = logits
        return outputs
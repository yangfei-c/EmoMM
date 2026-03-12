import torch
from transformers import XLMRobertaForSequenceClassification


class XLMRobertaVA(XLMRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.sigmoid = self.customHardSigmoid

    def customHardSigmoid(self, x):
        return torch.nn.functional.hardsigmoid(3 * x)

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
        # 调用父类的 forward
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=None,  # 我们自己计算 Loss，所以这里不传 label 给父类
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.sigmoid(outputs.logits)
        outputs.logits = logits
        return outputs
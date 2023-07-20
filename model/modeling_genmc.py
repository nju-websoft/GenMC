#coding=utf-8
from transformers import T5ForConditionalGeneration
from transformers.file_utils import ModelOutput
import torch
from .modeling_transformer import MyTransformer, SemanticMatch
from torch import nn

device = torch.device("cuda:0")
class GenMC(nn.Module):
    def __init__(self, model_path, num_hidden_layers, alpha, beta):
        super(GenMC, self).__init__()
        self.alpha = alpha
        self.beta = beta

        device = torch.device("cuda:0")
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
        dim = self.t5_model.config.d_model
        self.option_linear = nn.Linear(dim, 1).to(device)
        self.option_linear.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.semantic_matching = SemanticMatch(dim, num_hidden_layers).to(device)
        self.semantic_matching.device = device
        num_attention_heads = dim // 64
        self.transformer_laryer_de = MyTransformer(dim, num_attention_heads, num_hidden_layers).to(device)
        self.transformer_laryer_de.device = device
        self.relu = nn.ReLU()

        n_gpu = torch.cuda.device_count()
        layer_num = self.t5_model.config.num_layers
        layer_per_gpu = layer_num // n_gpu
        layer_per_gpu_remainder = layer_num % n_gpu
        device_map = {}
        cur_layer = 0
        for n in range(n_gpu):
            device_map[n] = []
            if n < layer_per_gpu_remainder:
                layer_assigned = layer_per_gpu+1
            else:
                layer_assigned = layer_per_gpu

            for i in range(layer_assigned):
                device_map[n].append(cur_layer)
                cur_layer += 1
        self.t5_model.parallelize(device_map)


    def forward(self, q_ids, q_mask, qo_ids, qo_mask, choice_num, clue_ids=None, answers=None):
        self.choice_num = choice_num
        if answers is not None and clue_ids is not None:
            opt_score, output_sequences = self.get_option_score(q_ids, q_mask, qo_ids, qo_mask)
            local_device = self.t5_model.device
            t5_output = self.t5_model(input_ids=q_ids.to(local_device), attention_mask=q_mask.to(local_device),
                                      labels=clue_ids.to(local_device))
            loss_ans = t5_output.loss
            loss = self.criterion(opt_score, answers)
            return self.alpha * loss + self.beta * loss_ans
        else:
            opt_score, output_sequences = self.get_option_score(q_ids, q_mask, qo_ids, qo_mask)
            return opt_score, output_sequences

    def get_option_score(self, q_ids, q_mask, qo_ids, qo_mask):
        local_device = self.t5_model.encoder.device
        t5_output = self.t5_model.encoder(input_ids=qo_ids.to(local_device), attention_mask=qo_mask.to(local_device))
        encoder_qo = t5_output[0]

        t5_output = self.t5_model.encoder(input_ids=q_ids.to(local_device), attention_mask=q_mask.to(local_device))
        encoder_q = t5_output[0]
        local_device = self.t5_model.device
        t5_output = self.t5_model.generate(
            encoder_outputs=ModelOutput(last_hidden_state=encoder_q.to(local_device)),
            attention_mask=q_mask.to(local_device),
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        output_sequences = t5_output.sequences
        output_sequences = output_sequences[:, 1:].contiguous()
        decoder_o = t5_output.decoder_hidden_states
        decoder_o = [item[-1] for item in decoder_o]
        decoder_o = torch.cat(decoder_o, dim=1)

        output_sequences_mask1 = output_sequences != 0
        output_sequences_mask2 = output_sequences != 1
        output_sequences_mask = output_sequences_mask1 * output_sequences_mask2
        output_sequences_mask = output_sequences_mask.long()
        decoder_qo = torch.cat([encoder_q, decoder_o], dim=1)
        output_sequences_mask = torch.cat([q_mask, output_sequences_mask], dim=1)
        local_device = self.transformer_laryer_de.device
        decoder_qo, _ = self.transformer_laryer_de(decoder_qo.to(local_device), output_sequences_mask.to(local_device))
        output_sequences_mask_ex = output_sequences_mask.unsqueeze(dim=1)
        output_sequences_mask_ex = output_sequences_mask_ex.expand(
            [output_sequences_mask_ex.size(0), self.choice_num, output_sequences_mask_ex.size(-1)]).contiguous()
        output_sequences_mask_ex = output_sequences_mask_ex.view(-1, output_sequences_mask.size(-1))
        decoder_qo = decoder_qo.unsqueeze(dim=1)
        decoder_qo = decoder_qo.expand(
            [decoder_qo.size(0), self.choice_num, decoder_qo.size(-2), decoder_qo.size(-1)]).contiguous()
        decoder_qo = decoder_qo.view(-1, decoder_qo.size(-2), decoder_qo.size(-1))
        local_device = self.semantic_matching.device
        semantic_vec, _, _ = self.semantic_matching(encoder_qo.to(local_device), decoder_qo.to(local_device),
                                                    qo_mask.to(local_device), output_sequences_mask_ex.to(local_device))
        local_device = self.option_linear.device
        opt_score = self.option_linear(semantic_vec.to(local_device)).view(-1, self.choice_num)

        return opt_score, output_sequences

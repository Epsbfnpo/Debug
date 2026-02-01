import math
import torch
import torch.nn as nn


class FineGrainedPromptTuning(torch.nn.Module):
    # [修改点 1]: __init__ 增加 selection_ratio 参数
    def __init__(self, side_encoder, fusion_module, layer_indices=None, selection_ratio=0.2):
        super().__init__()
        self.side_encoder = side_encoder
        self.fusion_module = fusion_module
        self.layer_indices = layer_indices
        self.selection_ratio = selection_ratio  # 保存筛选比例，例如 0.2 表示保留 20%

    # [修改点 2]: 新增 Token 筛选函数
    def token_selection(self, key_states, value_states, importance_map):
        """
        key_states: [Layers, Batch, Seq_Len, Dim] - 待筛选的特征
        importance_map: [Batch, Heads, Seq_Len, Seq_Len] - 最后一层的 Attention Map
        """
        # 1. 计算重要性分数
        # 取出 [CLS] token 对其他 token 的注意力 (Seq_Len 中的第 0 个是 CLS)
        # importance_map shape: [B, H, N, N]
        # 我们关注 CLS (index 0) 对其他 tokens (index 1:) 的关注度
        cls_attention = importance_map[:, :, 0, 1:]  # [B, H, N-1]

        # 对所有 Head 求平均
        token_scores = cls_attention.mean(dim=1)  # [B, N-1]

        # 2. 确定要保留的 Token 数量 K
        N = token_scores.shape[1]
        K = int(N * self.selection_ratio)
        if K < 1: K = 1  # 至少保留 1 个

        # 3. 获取 Top-K 的索引
        # values, indices: [B, K]
        _, topk_indices = torch.topk(token_scores, K, dim=1)

        # 因为我们之前去掉了 CLS (index 0)，所以现在的索引是相对于 1:N 的
        # 如果需要对应回原始 key_states，索引需要 +1 (因为 key_states 包含 CLS)
        # 但通常 FPT 只融合 spatial tokens，所以我们直接提取 spatial tokens

        # 4. 准备 gather
        # key_states shape: [L, B, N_total, D]
        # 我们只处理 spatial tokens (去掉 CLS)，所以先切片
        key_states_spatial = key_states[:, :, 1:, :]  # [L, B, N-1, D]
        value_states_spatial = value_states[:, :, 1:, :]  # [L, B, N-1, D]

        # 扩展 indices 以便 gather
        # indices shape [B, K] -> need to broadcast to [L, B, K, D]
        # 这比较麻烦，我们换一种方式：遍历 Batch (虽然慢一点点，但逻辑清晰且不容易出错)
        # 或者利用 torch.gather 的特性

        L, B, _, D = key_states_spatial.shape

        # 调整 indices 维度: [B, K] -> [1, B, K, 1] -> expand to [L, B, K, D]
        gather_indices = topk_indices.view(1, B, K, 1).expand(L, B, K, D)

        # 执行 gather
        selected_keys = torch.gather(key_states_spatial, 2, gather_indices)
        selected_values = torch.gather(value_states_spatial, 2, gather_indices)

        return selected_keys, selected_values

    # [修改点 3]: forward 增加 attention_map 参数
    def forward(self, x_coarse, key_states, value_states, attention_map=None, return_features=False):
        if self.layer_indices is not None:
            key_states = key_states[self.layer_indices]
            value_states = value_states[self.layer_indices]

        # [修改点 4]: 调用筛选逻辑
        if attention_map is not None and self.selection_ratio < 1.0:
            key_states, value_states = self.token_selection(key_states, value_states, attention_map)

        fine_grained_states = self.fusion_module(key_states, value_states)

        out = self.side_encoder(
            x_coarse,
            fine_grained_states,
            interpolate_pos_encoding=True,
            return_features=return_features
        )

        if return_features:
            return out

        return out.logits


class FusionModule(torch.nn.Module):
    def __init__(self, num_layers, in_dim, out_dim, num_heads, num_prompts,
                 prompt_dim=None, prompt_norm=True, prompt_proj=False, p_dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_prompts = num_prompts
        self.head_size = in_dim // num_heads

        self.prompt_dim = in_dim if prompt_dim is None else prompt_dim
        assert prompt_proj or self.prompt_dim == in_dim, 'Prompt projection is required when prompt dimension is different from input dimension.'

        self.prompts = nn.Parameter(torch.zeros(num_layers, 1, num_prompts, self.prompt_dim))
        self.layer_norm = nn.LayerNorm(self.prompt_dim) if prompt_norm else nn.Identity()
        self.prompt_proj = nn.Linear(self.prompt_dim, in_dim) if prompt_proj else nn.Identity()

        self.out_proj = torch.nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(p_dropout)

        nn.init.normal_(self.prompts, std=1e-6)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 1, 3, 2, 4)

    def forward(self, key_layer, value_layer):
        assert key_layer.shape[0] == self.num_layers, f'Key layer mismatch: got {key_layer.shape[0]}, expected {self.num_layers}'

        prompts = self.prompts.expand(-1, key_layer.shape[1], -1, -1)
        query_layer = self.prompt_proj(self.layer_norm(prompts))

        query_layer = self.transpose_for_scores(query_layer)

        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 1, 3, 2, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.in_dim,)
        context_layer = context_layer.view(new_context_layer_shape)

        if self.prompt_dim == self.in_dim:
            context_layer = context_layer + prompts
            context_layer = self.out_proj(context_layer)
        else:
            context_layer = self.out_proj(context_layer)
            context_layer = context_layer + prompts

        context_layer = self.dropout(context_layer)
        return context_layer
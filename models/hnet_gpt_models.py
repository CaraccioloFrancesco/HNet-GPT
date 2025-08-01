# hnet_gpt_models.py
"""Standalone model definitions for HNet-GPT research"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config




def create_pure_gpt2_baseline(vocab_size):
    """Pure GPT-2 baseline for comparison"""

    class PureGPT2(nn.Module):
        def __init__(self, vocab_size, embed_dim=768):
            super().__init__()

            # Create GPT-2 configuration
            self.config = GPT2Config(
                vocab_size=vocab_size,
                n_embd=embed_dim,
                n_layer=12,
                n_head=12,
                activation_function='gelu_new',
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
            )

            # Create GPT-2 model
            self.gpt2 = GPT2Model(self.config)

            # Output projection
            self.output = nn.Linear(embed_dim, vocab_size)

        def forward(self, input_ids, labels=None):
            # Forward through GPT-2
            outputs = self.gpt2(input_ids)
            hidden_states = outputs.last_hidden_state

            # Project to vocabulary
            logits = self.output(hidden_states)

            # Compute loss
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            return logits, loss

            
        def generate(self, input_ids, max_length=None, num_return_sequences=1, temperature=1.0, do_sample=False, **kwargs):
            """FIXED: Use parameter instead of tokenizer reference"""
            self.eval()
            device = input_ids.device

            if max_length is None:
                max_length = input_ids.size(1) + 50

            max_length = min(max_length, 512)
            generated = input_ids.clone()

            top_k = kwargs.get('top_k', 50)
            top_p = kwargs.get('top_p', 1.0)
            repetition_penalty = kwargs.get('repetition_penalty', 1.2)
            eos_token_id = kwargs.get('eos_token_id', 50256)  # Get EOS token from kwargs

            with torch.no_grad():
                for _ in range(max_length - input_ids.size(1)):
                    logits, _ = self.forward(generated)
                    next_token_logits = logits[:, -1, :]

                    # Apply repetition penalty
                    if repetition_penalty != 1.0 and generated.size(1) > 0:
                        for i in range(generated.size(0)):
                            for prev_token_idx in set(generated[i, max(0, generated.size(1) - 10):].tolist()):
                                if prev_token_idx < next_token_logits.size(-1):
                                    if next_token_logits[i, prev_token_idx] < 0:
                                        next_token_logits[i, prev_token_idx] *= repetition_penalty
                                    else:
                                        next_token_logits[i, prev_token_idx] /= repetition_penalty

                    next_token_logits = next_token_logits / max(temperature, 1e-8)

                    if do_sample:
                        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

                        all_filtered_out = (filtered_logits == -float('Inf')).all(dim=-1)
                        if all_filtered_out.any():
                            for i in range(all_filtered_out.size(0)):
                                if all_filtered_out[i]:
                                    filtered_logits[i, 0] = 1.0

                        filtered_logits = torch.nan_to_num(filtered_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
                        probs = F.softmax(filtered_logits, dim=-1)
                        probs = torch.clamp(probs, min=1e-9)
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    generated = torch.cat([generated, next_token], dim=-1)

                    # Use eos_token_id parameter instead of tokenizer
                    if eos_token_id is not None and (next_token == eos_token_id).any():
                        break

                    if generated.size(1) >= input_ids.size(1) + 3:
                        if (generated[:, -1] == generated[:, -2]).any() and \
                          (generated[:, -1] == generated[:, -3]).any():
                            break

            return generated


    return PureGPT2(vocab_size)




def create_pure_hnet_model(vocab_size, tokenizer):
    """
    Create Pure HNet model - End-to-end hierarchical transformer

    Architecture Philosophy:
    - Multi-level hierarchical processing (chunk → global → sequence)
    - Attention-based chunk pooling for better representations
    - Gated fusion between hierarchical and sequential features
    - Designed specifically for structured text like code
    """

    class PureHNetModel(nn.Module):
        def __init__(self, vocab_size, tokenizer, embed_dim=768):
            super().__init__()
            self.tokenizer = tokenizer
            self.embed_dim = embed_dim

            print(f"Building Pure HNet Architecture:")
            print(f"Vocabulary: {vocab_size:,} tokens")
            print(f"Embedding dimension: {embed_dim}")

            # Core Embeddings 
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.pos_embed = nn.Embedding(1024, embed_dim)
            self.dropout = nn.Dropout(0.1)

            # Hierarchical Parameters 
            self.num_chunks = 32  # Number of hierarchical chunks
            self.chunk_size = 512 // self.num_chunks
            self.chunk_overlap = int(self.chunk_size * 0.25)  # 25% overlap

            print(f"Chunking strategy: {self.num_chunks} chunks of size {self.chunk_size}")
            print(f"Chunk overlap: {self.chunk_overlap} tokens")

            # Level 1: Chunk-Level Processing 
            chunk_encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.chunk_encoder = nn.TransformerEncoder(chunk_encoder_layer, num_layers=6)

            # Level 2: Global Context Processing 
            global_encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.global_encoder = nn.TransformerEncoder(global_encoder_layer, num_layers=4)

            # Level 3: Hierarchical-to-Sequential Bridge 
            self.chunk_projection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim)
            )

            # Gating mechanism for hierarchical fusion
            self.hierarchical_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(0.1),
                nn.Sigmoid()
            )

            # Level 4: Final Sequence Processing 
            sequence_encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.sequence_encoder = nn.TransformerEncoder(sequence_encoder_layer, num_layers=6)

            # Output Layer 
            self.output = nn.Linear(embed_dim, vocab_size)

            # Initialize weights
            self.apply(self._init_weights)

            total_params = sum(p.numel() for p in self.parameters())
            print(f"Total parameters: {total_params:,}")

        def _init_weights(self, module):
            """Initialize model weights with small random values"""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

        def _create_hierarchical_chunks(self, hidden_states, input_ids):
            """
            Create hierarchical chunks with smart overlapping

            Key Innovation: Attention-based pooling instead of mean pooling
            for better chunk representations
            """
            B, L, D = hidden_states.shape
            chunks = []

            for batch_idx in range(B):
                batch_chunks = []

                # Create overlapping chunks for better context
                for i in range(self.num_chunks):
                    start = i * self.chunk_size
                    end = min(start + self.chunk_size + self.chunk_overlap, L)

                    if start < L:
                        chunk_tokens = hidden_states[batch_idx, start:end]
                        # Use attention pooling for better representations
                        chunk_repr = self._attention_pool(chunk_tokens)
                        batch_chunks.append(chunk_repr)

                # Ensure consistent chunk count
                while len(batch_chunks) < self.num_chunks:
                    batch_chunks.append(torch.zeros(D, device=hidden_states.device))

                batch_chunks = batch_chunks[:self.num_chunks]
                chunks.append(torch.stack(batch_chunks))

            return torch.stack(chunks, dim=0)

        def _attention_pool(self, chunk_tokens):
            """
            Attention-based pooling for chunk representation
            Better than mean pooling as it focuses on important tokens
            """
            if chunk_tokens.size(0) == 0:
                return torch.zeros(self.embed_dim, device=chunk_tokens.device)

            # Compute attention weights based on token importance
            chunk_mean = chunk_tokens.mean(dim=0, keepdim=True)
            attention_scores = torch.sum(chunk_tokens * chunk_mean, dim=-1)
            attention_weights = torch.softmax(attention_scores, dim=0)

            # Weighted sum using attention
            return torch.sum(chunk_tokens * attention_weights.unsqueeze(-1), dim=0)

        def forward(self, input_ids, labels=None):
            """
            Four-level hierarchical forward pass:
            1. Embeddings + Chunking
            2. Chunk-level processing
            3. Global context processing
            4. Hierarchical-sequential fusion
            5. Final sequence processing
            """
            B, L = input_ids.shape

            #  Level 1: Embeddings 
            positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
            hidden_states = self.embed(input_ids) + self.pos_embed(positions)
            hidden_states = self.dropout(hidden_states)

            # Level 2: Chunk Processing 
            chunks = self._create_hierarchical_chunks(hidden_states, input_ids)
            encoded_chunks = self.chunk_encoder(chunks)

            # Level 3: Global Context 
            global_context = self.global_encoder(encoded_chunks)

            # Level 4: Hierarchical-Sequential Bridge
            # Project hierarchical features back to sequence space
            projected_chunks = self.chunk_projection(global_context)

            # Expand chunks back to sequence length
            chunk_expanded = torch.repeat_interleave(projected_chunks, self.chunk_size, dim=1)
            if chunk_expanded.size(1) > L:
                chunk_expanded = chunk_expanded[:, :L, :]
            elif chunk_expanded.size(1) < L:
                padding = torch.zeros(B, L - chunk_expanded.size(1), self.embed_dim,
                                    device=hidden_states.device)
                chunk_expanded = torch.cat([chunk_expanded, padding], dim=1)

            # Gated fusion between hierarchical and sequential features
            gate_input = torch.cat([hidden_states, chunk_expanded], dim=-1)
            gate = self.hierarchical_gate(gate_input)
            combined = gate * chunk_expanded + (1 - gate) * hidden_states

            # Level 5: Final Sequence Processing 
            final_hidden = self.sequence_encoder(combined)

            # Output projection
            logits = self.output(final_hidden)

            # Compute loss if labels provided
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            return logits, loss

        def generate(self, input_ids, max_length=None, num_return_sequences=1, temperature=1.0, do_sample=False, **kwargs):
            """Robust generation method matching other models"""
            self.eval()
            device = input_ids.device

            if max_length is None:
                max_length = input_ids.size(1) + 50

            max_length = min(max_length, 512)
            generated = input_ids.clone()

            top_k = kwargs.get('top_k', 50)
            top_p = kwargs.get('top_p', 1.0)
            repetition_penalty = kwargs.get('repetition_penalty', 1.2)

            with torch.no_grad():
                for _ in range(max_length - input_ids.size(1)):
                    logits, _ = self.forward(generated)
                    next_token_logits = logits[:, -1, :]

                    # Apply repetition penalty
                    if repetition_penalty != 1.0 and generated.size(1) > 0:
                        for i in range(generated.size(0)):
                            for prev_token_idx in set(generated[i, max(0, generated.size(1) - 10):].tolist()):
                                if prev_token_idx < next_token_logits.size(-1):
                                    if next_token_logits[i, prev_token_idx] < 0:
                                        next_token_logits[i, prev_token_idx] *= repetition_penalty
                                    else:
                                        next_token_logits[i, prev_token_idx] /= repetition_penalty

                    next_token_logits = next_token_logits / max(temperature, 1e-8)

                    if do_sample:
                        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

                        all_filtered_out = (filtered_logits == -float('Inf')).all(dim=-1)
                        if all_filtered_out.any():
                            for i in range(all_filtered_out.size(0)):
                                if all_filtered_out[i]:
                                    filtered_logits[i, 0] = 1.0

                        filtered_logits = torch.nan_to_num(filtered_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
                        probs = F.softmax(filtered_logits, dim=-1)
                        probs = torch.clamp(probs, min=1e-9)
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    generated = torch.cat([generated, next_token], dim=-1)

                    # Stopping conditions
                    if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                        if (next_token == self.tokenizer.eos_token_id).any():
                            break

                    if generated.size(1) >= input_ids.size(1) + 3:
                        if (generated[:, -1] == generated[:, -2]).any() and \
                          (generated[:, -1] == generated[:, -3]).any():
                            break

            return generated

    return PureHNetModel(vocab_size, tokenizer)




def create_hnet_gpt2_hybrid(vocab_size, tokenizer):
    """HNet encoder with GPT-2 decoder blocks"""

    class HNetGPT2Hybrid(nn.Module):
        def __init__(self, vocab_size, tokenizer, embed_dim=768, num_chunks=24):
            super().__init__()
            self.tokenizer = tokenizer  # Store as instance variable

            # Use GPT-2's embedding dimension for compatibility
            self.embed_dim = embed_dim

            # Embeddings (matching GPT-2's setup)
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.pos_embed = nn.Embedding(1024, embed_dim)
            self.dropout = nn.Dropout(0.1)

            # HNet Encoder (from original) 
            self.num_chunks = 32  # Smaller chunks for better granularity
            self.chunk_size = 512 // self.num_chunks
            self.chunk_overlap = int(self.chunk_size * 0.25)  # 25% overlap


            # Hierarchical chunk encoder (from HNet)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,  # Match GPT-2's attention heads
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.chunk_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

            # GPT-2 Decoder Blocks 
            # Load GPT-2 configuration and extract decoder blocks
            gpt2_config = GPT2Config(
                vocab_size=vocab_size,
                n_embd=embed_dim,
                n_layer=12,  # Use 12 layers like GPT-2 small
                n_head=12,
                activation_function='gelu_new',
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
            )

            # Create GPT-2 model and extract transformer blocks
            gpt2_model = GPT2Model(gpt2_config)
            self.gpt2_blocks = gpt2_model.h  # Extract the transformer blocks
            self.ln_f = gpt2_model.ln_f  # Final layer norm


            self.hierarchical_projection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, embed_dim)
            )

            # Add dropout to gating:
            self.hierarchical_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(0.1),  # Add dropout here
                nn.Sigmoid()
            )

            # Initialize bias to favor original representations
            nn.init.constant_(self.hierarchical_gate[0].bias, -2.0)


            # Output projection
            self.output = nn.Linear(embed_dim, vocab_size)

            # Initialize weights
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()


        def _create_adaptive_chunks(self, hidden_states, input_ids):
            """Create chunks based on code structure boundaries"""
            B, L, D = hidden_states.shape
            chunks = []

            for batch_idx in range(B):
                # Decode tokens back to text to find structure
                try:
                    # Get non-padded tokens
                    tokens = input_ids[batch_idx]
                    text = self.tokenizer.decode(tokens, skip_special_tokens=True)

                    # Find function/class boundaries using simple heuristics
                    boundaries = [0]
                    lines = text.split('\n')
                    current_pos = 0

                    for line in lines:
                        if line.strip().startswith(('def ', 'class ', 'if ', 'for ', 'while ')):
                            # Convert line position back to token position (approximation)
                            char_pos = text.find(line)
                            if char_pos > 0:
                                # Rough token position estimate
                                token_pos = min(int(char_pos * 0.3), L-1)  # Rough char-to-token ratio
                                if token_pos > boundaries[-1] + 10:  # Minimum chunk size
                                    boundaries.append(token_pos)

                    boundaries.append(L)

                except:
                    # Fallback to fixed chunking if parsing fails
                    boundaries = [i * (L // self.num_chunks) for i in range(self.num_chunks + 1)]
                    boundaries[-1] = L

                # Create chunks from boundaries
                batch_chunks = []
                for i in range(len(boundaries) - 1):
                    start, end = boundaries[i], boundaries[i + 1]

                    # Add overlap for non-first chunks
                    if i > 0:
                        start = max(0, start - self.chunk_overlap)

                    # Add overlap for non-last chunks
                    if i < len(boundaries) - 2:
                        end = min(L, end + self.chunk_overlap)

                    if end > start:
                        chunk_tokens = hidden_states[batch_idx, start:end]
                        chunk_repr = torch.mean(chunk_tokens, dim=0)
                        batch_chunks.append(chunk_repr)

                # Pad to consistent number of chunks
                while len(batch_chunks) < self.num_chunks:
                    batch_chunks.append(torch.zeros(D, device=hidden_states.device))

                # Take first num_chunks if we have too many
                batch_chunks = batch_chunks[:self.num_chunks]
                chunks.append(torch.stack(batch_chunks))
            
            return chunks  # Chunks is already a list of tensors for each batch



        def forward(self, input_ids, labels=None):
            B, L = input_ids.shape

            # Embeddings 
            positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
            hidden_states = self.embed(input_ids) + self.pos_embed(positions)
            hidden_states = self.dropout(hidden_states)

            # HNet Hierarchical Encoding
            # Create adaptive chunks based on code structure
            chunks = self._create_adaptive_chunks(hidden_states, input_ids)
            chunk_tensor = torch.stack(chunks, dim=0)  # dim=0 for batch dimension
            encoded_chunks = self.chunk_encoder(chunk_tensor)


            # GPT-2 Decoder with Cross-Attention 
            # Expand chunk representations back to sequence length
            chunk_expanded = torch.repeat_interleave(encoded_chunks, self.chunk_size, dim=1)
            if chunk_expanded.size(1) > L:
                chunk_expanded = chunk_expanded[:, :L, :]
            elif chunk_expanded.size(1) < L:
                padding = torch.zeros(B, L - chunk_expanded.size(1), self.embed_dim, device=hidden_states.device)
                chunk_expanded = torch.cat([chunk_expanded, padding], dim=1)

            # Project hierarchical features
            hierarchical_features = self.hierarchical_projection(chunk_expanded)

            # Learned gating
            gate_input = torch.cat([hidden_states, hierarchical_features], dim=-1)
            gate = self.hierarchical_gate(gate_input)

            # With this (residual scaling):
            gated_hierarchical = gate * hierarchical_features + (1 - gate) * hidden_states

            complexity_score = torch.mean(gate, dim=-1, keepdim=True)  # [B, L, 1]
            alpha = 0.8 + 0.2 * complexity_score  # Dynamic 0.8-1.0 range
            hidden_states = alpha * hidden_states + (1 - alpha) * gated_hierarchical


            # Apply GPT-2 transformer blocks
            for block in self.gpt2_blocks:
                outputs = block(hidden_states)
                hidden_states = outputs[0]

            # Final layer norm
            hidden_states = self.ln_f(hidden_states)

            # Output projection
            logits = self.output(hidden_states)

            # Compute loss
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            return logits, loss

            
        def generate(self, input_ids, max_length=None, num_return_sequences=1, temperature=1.0, do_sample=False, **kwargs):
          
            self.eval()
            device = input_ids.device

            if max_length is None:
                max_length = input_ids.size(1) + 50

            max_length = min(max_length, 512)
            generated = input_ids.clone()

            top_k = kwargs.get('top_k', 50)
            top_p = kwargs.get('top_p', 1.0)
            repetition_penalty = kwargs.get('repetition_penalty', 1.2)

            with torch.no_grad():
                for _ in range(max_length - input_ids.size(1)):
                    logits, _ = self.forward(generated)
                    next_token_logits = logits[:, -1, :]

                    # Apply repetition penalty
                    if repetition_penalty != 1.0 and generated.size(1) > 0:
                        for i in range(generated.size(0)):
                            for prev_token_idx in set(generated[i, max(0, generated.size(1) - 10):].tolist()):
                                if prev_token_idx < next_token_logits.size(-1):
                                    if next_token_logits[i, prev_token_idx] < 0:
                                        next_token_logits[i, prev_token_idx] *= repetition_penalty
                                    else:
                                        next_token_logits[i, prev_token_idx] /= repetition_penalty

                    next_token_logits = next_token_logits / max(temperature, 1e-8)

                    if do_sample:
                        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

                        all_filtered_out = (filtered_logits == -float('Inf')).all(dim=-1)
                        if all_filtered_out.any():
                            for i in range(all_filtered_out.size(0)):
                                if all_filtered_out[i]:
                                    filtered_logits[i, 0] = 1.0

                        filtered_logits = torch.nan_to_num(filtered_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
                        probs = F.softmax(filtered_logits, dim=-1)
                        probs = torch.clamp(probs, min=1e-9)
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    generated = torch.cat([generated, next_token], dim=-1)

                    # FIXED: Use self.tokenizer instead of tokenizer
                    if self.tokenizer.eos_token_id is not None and (next_token == self.tokenizer.eos_token_id).any():
                        break

                    if generated.size(1) >= input_ids.size(1) + 3:
                        if (generated[:, -1] == generated[:, -2]).any() and \
                          (generated[:, -1] == generated[:, -3]).any():
                            break

            return generated

    return HNetGPT2Hybrid(vocab_size, tokenizer)

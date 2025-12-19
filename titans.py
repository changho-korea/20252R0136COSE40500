
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralMemory(nn.Module):
    """
    Neural Long-term Memory Module.
    Learns to memorize historical context at test time by updating its weights
    based on a 'surprise' metric (gradient of associative memory loss).
    """
    def __init__(self, input_dim, memory_dim, memory_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        
        # Projections for Key, Value, Query (for retrieval)
        # Using depth-wise separable convolution as suggested in the paper 
        # (simplified here as linear for core logic first, can add conv later if needed for performance/exact match)
        # Paper mentions: 1D depthwise-separable convolution layer after each of the Q/K/V projections.
        
        self.feature_proj = nn.Linear(input_dim, input_dim * 3) # Combined Q, K, V for inputs
        
        # The Memory Model M_t.
        # Paper says: "We focus on simple MLPs with L_M >= 1 layers"
        # M_t parameters are what gets updated.
        # Ideally, we want to maintain the STATE of these parameters.
        # Since standard nn.Module parameters are optimized by the outer optimizer, 
        # we need to be careful. The M_t weights are HIDDEN STATES in this RNN view.
        # So we shouldn't define M as a standard sub-module with parameters that PyTorch optimizer sees directly 
        # as global parameters, UNLESS we treat them as initial values.
        # But for an RNN, the hidden state is initialized usually to zeros or learned initial state.
        
        self.memory_layers = memory_layers
        self.act = nn.SiLU()

        # Learnable parameters for the update rule (Decay and Momentum factor)
        # These can be data dependent or chunk dependent.
        # Paper eq 10: "eta_t is a data-dependent surprise decay... theta_t is controlling momentary surprise"
        # We project input to get eta_t and theta_t
        self.gate_proj = nn.Linear(input_dim, input_dim * 3) # For eta (decay), theta (lr), alpha (forget)
        
        # Initial memory weights. The memory function M maps Key -> Value.
        # So input size is memory_dim (key size) and output is memory_dim (value size).
        # We assume key and value dims are same as input_dim for simplicity unless specified.
        # The paper implies K and V are projected from X.
        
        # Let's define the shape of the Memory weights.
        # If M is an MLP, it has weights W1, b1, W2, b2...
        # We need to store these as a "Hidden State" tensor.
        # A simple linear memory: W in R^{d x d}.
        # MLP memory: list of tensors.
        
        # For efficiency in this recurrent implementation, let's assume M is a Linear model first (L_M=1)
        # or a 2-layer MLP. 
        # The paper emphasizes "Deep Memory" (L_M >= 2) is better.
        
        # We need a meta-parameter for the INITIAL state of the memory (M_0).
        # We can make it learnable.
        
        # To keep it simple and runnable:
        # We'll treat the weights of M as a single large vector or manage them manualy.
        # Let's simplify: M is small MLP.
        pass

    def forward(self, x, memory_state=None):
        pass

class SimpleNeuralMemory(nn.Module):
    """
    A simplified implementation of NeuralMemory where M is a Linear layer (Matrix Memory).
    This computes equivalent to linear attention variants but with the specific surprise-based update.
    Paper Eq 8-14.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wq = nn.Linear(dim, dim, bias=False) # For retrieval
        
        # Gating network: produces alpha (forget), eta (momentum decay), theta (learning rate)
        # All in [0, 1] usually.
        self.gate_net = nn.Linear(dim, 3) 

    def forward(self, x, state=None):
        """
        x: [batch, seq_len, dim]
        state: Tuple (M, S) corresponding to Memory and Momentum buffers.
               M is [batch, dim, dim] (since it maps dim->dim)
               S is [batch, dim, dim]
        """
        b, s, d = x.shape
        
        k = self.wk(x) # [b, s, d]
        v = self.wv(x) # [b, s, d]
        q = self.wq(x) # [b, s, d]
        
        # Gates
        # alpha uses sigmoid? Paper says alpha in [0,1].
        # eta, theta also need range constraints.
        gates = torch.sigmoid(self.gate_net(x)) # [b, s, 3]
        alpha = gates[..., 0].view(b, s, 1, 1) # scalar per step per batch usually, or elementwise?
        # Paper says alpha_t in [0,1]. Usually scalar or vector. 
        # If we want element-wise control, we can do [b,s,1] or [b,s,d].
        # Let's assume scalar per token for simplicity as implied by "proportion of memory size".
        # Actually Eq 13: M_t = (1-alpha_t)M_{t-1} + S_t. 
        # If alpha is scalar, it decays everything. If vector, elementwise.
        # Let's try scalar gating first (shared across dimensions) or maybe per-channel.
        # Paper mentions "gating mechanism to forget...". 
        
        eta = gates[..., 1].view(b, s, 1, 1)
        theta = gates[..., 2].view(b, s, 1, 1)
        
        # Helper for scalar expansion
        def expand(t): return t

        if state is None:
            # Initialize M and S
            # M_0, S_0
            M = torch.zeros(b, d, d, device=x.device)
            S = torch.zeros(b, d, d, device=x.device)
        else:
            M, S = state

        outputs = []
        
        # Sequential Processing (Recurrent)
        # Can be parallelized with prefix scan if linearized, but let's do loop for clarity/correctness first.
        for t in range(s):
            xt = x[:, t, :] # [b, d]
            kt = k[:, t, :] # [b, d]
            vt = v[:, t, :] # [b, d]
            qt = q[:, t, :] # [b, d]
            
            at = alpha[:, t, :, :]
            et = eta[:, t, :, :]
            tt = theta[:, t, :, :]
            
            # 1. Retrieve current memory (Observation/Prediction before update? or Inference only?)
            # Paper Eq 15: y_t = M*(q_t)  (forward pass without weight update)
            # We can use M_{t-1} to answer query at t? Or M_t? context suggests using memory to answer.
            # Usually in RNNs, y_t depends on h_t. In Titans "We retrieve the past information... ht = Mt-1(qt)" 
            # (Eq 21 in MAC). So use previous memory.
            
            # M acts as a linear map: y = x W_M (or similar)
            # If M is matrix [d, d], then y = q @ M or M @ q.
            # Loss is || M(k) - v ||^2. If M is [d,d], usually v = M k or v = k M.
            # Let's assume v = k @ M. Then grad_M = k^T (k@M - v).
            
            # Retrieval
            yt = torch.bmm(qt.unsqueeze(1), M).squeeze(1) # [b, 1, d] @ [b, d, d] -> [b, 1, d]
            outputs.append(yt)
            
            # 2. Update Step
            # Loss L = || k @ M_{t-1} - v ||^2
            # Recalculate prediction with current memory for the *loss*
            pred = torch.bmm(kt.unsqueeze(1), M).squeeze(1) # k @ M
            error = pred - vt # [b, d]
            
            # Gradient wrt M:
            # dL/dM = 2 * k^T @ (k @ M - v) = 2 * k^T @ error
            # Shape: [b, d, 1] @ [b, 1, d] -> [b, d, d]
            grad = torch.bmm(kt.unsqueeze(2), error.unsqueeze(1)) 
            
            # Momentary Surprise (Eq 10): - theta * grad
            # Note: Paper says "measure surprise... with gradient". 
            # Eq 8: M = M - theta * grad.
            # Eq 10: S_t = eta * S_{t-1} - theta * grad
            
            # Why negative gradient? Gradient descent minimizes loss. 
            # If we want to "memorize", we want to minimize the reconstruction error of (k,v).
            # So updating in negative gradient direction makes M better at mapping k->v.
            
            momentary_surprise = - tt * grad
            
            # Update Momentum/Surprise State
            S = et * S + momentary_surprise
            
            # Update Memory (Eq 13)
            M = (1 - at) * M + S
            
        outputs = torch.stack(outputs, dim=1)
        return outputs, (M, S)

class PersistentMemory(nn.Module):
    def __init__(self, num_tokens, dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_tokens, dim))
        
    def forward(self, batch_size):
        return self.memory.unsqueeze(0).expand(batch_size, -1, -1)

class TitansMAC(nn.Module):
    """
    Titans: Memory as a Context.
    Segments input, uses NeuralMemory to compress past segments, 
    concatenates retrieved history + persistent memory + current segment,
    then runs attention.
    """
    def __init__(self, dim, memory_dim, segment_len, persistent_tokens=16):
        super().__init__()
        self.segment_len = segment_len
        self.neural_memory = SimpleNeuralMemory(dim)
        self.persistent_memory = PersistentMemory(persistent_tokens, dim)
        
        # Core Branch: Standard Transformer Encoder Layer or similar
        # For demonstration we use a simple MultiheadAttention block
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Projection for memory update input if needed.
        # Paper says: "We then use y_t (output of attn) to update the long-term memory... M_t = M_{t-1}(y_t)"
        # Wait, Eq 24: "M_t = M_{t-1}(y_t)". 
        # Does this mean we feed y_t as INPUT to the memory update step?
        # Eq 21: h_t = M_{t-1}(q_t) (Retrieve)
        # Eq 24: M_t = M_{t-1}(y_t) (Update state?)
        # 
        # Actually Eq 24 notation M(y) is ambiguous. "Update weights through forward pass"
        # It means we basically feed y_t into the memory module's "forward" (update mechanism).
        # So the input to NeuralMemory for the upgrade phase is y_t.
        # But NeuralMemory takes (k, v) or x -> (k,v).
        # Paper says: "We then use y_t to update... and the final output o_t = y_t + M*_t(y_t)" - Eq 25 is diff.
        
        # Let's stick to logical flow:
        # 1. Segment S_t comes in.
        # 2. Retrieve history: h_t = NeuralMemory.retrieve(S_t) (using M_{t-1})
        # 3. Construct context: C_t = [P, h_t, S_t]
        # 4. Attention: y_raw = Attn(C_t). We only care about output corresponding to S_t part?
        #    Paper Fig 3a shows attention is causal. 
        #    The output y_t corresponds to the S_t part of the sequence.
        # 5. Update Memory: Feed y_t into NeuralMemory to update M_{t-1} -> M_t.
        #    So y_t acts as the data 'x' for the memory module for this chunk.
        
    def forward(self, x):
        b, n, d = x.shape
        # Pad if needed
        if n % self.segment_len != 0:
            pad_len = self.segment_len - (n % self.segment_len)
            x = F.pad(x, (0, 0, 0, pad_len))
            n = x.shape[1]
            
        segments = x.chunk(n // self.segment_len, dim=1)
        
        memory_state = None # (M, S)
        final_outputs = []
        
        for seg in segments:
            # seg: [b, seg_len, d]
            
            # 1. Retrieve Historical Context
            # We use the segment as query to retrieve info from OLD memory
            # NeuralMemory forward does both retrieve and update?
            # We need to split operations potentially.
            # My SimpleNeuralMemory does both in one pass.
            # But here we need to retrieve BEFORE update, using the raw segment?
            # Equation 21: q_t = S(t) W_Q, h_t = M_{t-1}(q_t).
            # This is exactly the "Retrieval" step in my SimpleNeuralMemory using 'seg' as input,
            # BUT without updating the state yet.
            
            # Let's add a method or flag to NeuralMemory to just retrieve.
            # Or simpler: access the helper.
            
            # Extract q for retrieval from seg
            # We need access to internal Wq of memory module or expose a retrieve method.
            h_t = self.neural_memory.retrieve(seg, memory_state) # [b, seg_len, d]
            
            # 2. Persistent Memory
            p_mem = self.persistent_memory(b) # [b, p_len, d]
            
            # 3. Concatenate
            # Context = [P, h, S]
            # Wait, Eq 22: ~S(t) = [P || h_t || S(t)].
            # Size: p_len + seg_len + seg_len.
            context = torch.cat([p_mem, h_t, seg], dim=1)
            
            # 4. Attention
            # Causal mask? Paper says "Full causal attention in each window".
            # So within this context window, we mask.
            # But h_t is history (no causality constraint relative to S_t? h_t depends on past only).
            # P is fixed.
            # S_t is valid.
            # So standard causal mask on the whole thing?
            # Or is it "Cross attention"? No, "Input context... to the attention module".
            
            # MultiheadAttention need mask.
            # src_mask for [b, L, d].
            L = context.shape[1]
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            
            attn_out, _ = self.attn(context, context, context, attn_mask=mask, is_causal=True)
            
            # Take only the part corresponding to S(t)
            # The last seg_len tokens.
            y_t = attn_out[:, -self.segment_len:, :]
            y_t = self.norm1(y_t + seg) # Residual on the segment part? Or just y_t?
                                        # Usually transformer block is x + attn(norm(x)).
                                        # But here context structure is complex.
                                        # Let's assume standard post-norm or pre-norm res block on the segment part.
                                        # Eq 23: y_t = Attn(~S).
            
            # FFN
            y_t = y_t + self.ffn(self.norm2(y_t)) # Simple block
            
            final_outputs.append(y_t)
            
            # 5. Update Memory
            # Eq 24: M_t = M_{t-1}(y_t)
            # We use y_t (the processed output) to update the memory.
            # So we call neural_memory.forward(y_t, state=memory_state) which returns new state.
            # We ignore the output of this call (which would be retrieved info from M_t? or similar).
            # Actually Eq 25 says o_t = y_t @ M*_t(y_t).
            # "o_t = y_t \otimes M*_t(y_t)". \otimes is gating or combining.
            # We skip Eq 25 for now or implement as simple addition/gating?
            # "normalize outputs... followed by non-linearity".
            
            _, memory_state = self.neural_memory(y_t, memory_state)
            
        return torch.cat(final_outputs, dim=1)

# Adding 'retrieve' method to SimpleNeuralMemory
def retrieve(self, x, state):
    if state is None:
        # Zero memory
        return torch.zeros_like(x)
    M, _ = state
    # q = x Wq
    q = self.wq(x)
    # y = q @ M
    # y = q @ M
    y = torch.bmm(q, M)
    return y
    
SimpleNeuralMemory.retrieve = retrieve


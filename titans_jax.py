
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax

class NeuralMemory(nn.Module):
    dim: int
    
    def setup(self):
        self.wk = nn.Dense(self.dim, use_bias=False)
        self.wv = nn.Dense(self.dim, use_bias=False)
        self.wq = nn.Dense(self.dim, use_bias=False)
        self.gate_net = nn.Dense(3) # alpha, eta, theta

    def __call__(self, x, state=None):
        # x: [batch, seq_len, dim]
        # state: (M, S)
        
        b, s, d = x.shape
        
        k = self.wk(x)
        v = self.wv(x)
        q = self.wq(x)
        
        gates = nn.sigmoid(self.gate_net(x))
        alpha = gates[..., 0:1] # [b, s, 1]
        eta = gates[..., 1:2]
        theta = gates[..., 2:3]
        
        if state is None:
            M = jnp.zeros((b, d, d))
            S = jnp.zeros((b, d, d))
        else:
            M, S = state

        # Scan over sequence
        # We need to scan over the time dimension (axis 1)
        # Inputs to scan: (M, S), (k, v, q, alpha, eta, theta)
        
        # Transpose to [s, b, ... ] for scan convenience or keep batch inside
        # JAX scan usually scans over leading axis of xs.
        
        xs = (
            jnp.swapaxes(k, 0, 1), 
            jnp.swapaxes(v, 0, 1), 
            jnp.swapaxes(q, 0, 1),
            jnp.swapaxes(alpha, 0, 1), 
            jnp.swapaxes(eta, 0, 1), 
            jnp.swapaxes(theta, 0, 1)
        )
        
        def scan_fn(carry, inputs):
            M_prev, S_prev = carry
            kt, vt, qt, at, et, tt = inputs
            # qt: [b, d]
            # M_prev: [b, d, d]
            
            # 1. Retrieve
            # y_t = qt @ M_prev
            yt = jnp.einsum('bd,bdd->bd', qt, M_prev)
            
            # 2. Surprise / Update
            # Loss = || kt @ M - vt ||^2
            # We want grad wrt M.
            # Analytical grad: 2 * kt.T @ (kt @ M - vt)
            # Shapes: kt [b,d], vt [b,d]
            # (kt @ M - vt) -> pred_err [b, d]
            # kt.T @ pred_err -> [b, d, d] (outer product per batch item)
            
            pred = jnp.einsum('bd,bdd->bd', kt, M_prev)
            error = pred - vt
            grad = jnp.einsum('bi,bj->bij', kt, error)
            
            # Momentary surprise
            momentary_surprise = - tt[..., None] * grad # tt is [b, 1], grad is [b, d, d]
            
            # Update Momentum
            # et: [b, 1]
            S_new = et[..., None] * S_prev + momentary_surprise
            
            # Update Memory
            # at: [b, 1]
            M_new = (1.0 - at[..., None]) * M_prev + S_new
            
            return (M_new, S_new), yt

        (M_final, S_final), outputs = lax.scan(scan_fn, (M, S), xs)
        
        # outputs is [s, b, d]. Transpose back.
        outputs = jnp.swapaxes(outputs, 0, 1)
        
        return outputs, (M_final, S_final)
    
    def retrieve(self, x, state):
        # x: [batch, seq_len, dim]
        # state: (M, S)
        if state is None:
            return jnp.zeros_like(x)
        
        M, _ = state
        q = self.wq(x)
        # y = q @ M
        # q: [b, s, d], M: [b, d, d]
        y = jnp.einsum('bsd,bdd->bsd', q, M)
        return y

class PersistentMemory(nn.Module):
    num_tokens: int
    dim: int
    
    def setup(self):
        self.memory = self.param('memory', nn.initializers.normal(), (self.num_tokens, self.dim))
        
    def __call__(self, batch_size):
        return jnp.broadcast_to(self.memory[None, ...], (batch_size, self.num_tokens, self.dim))

class TitansMAC(nn.Module):
    dim: int
    memory_dim: int
    segment_len: int
    persistent_tokens: int = 16
    
    def setup(self):
        self.neural_memory = NeuralMemory(self.dim)
        self.persistent_memory = PersistentMemory(self.persistent_tokens, self.dim)
        
        self.norm1 = nn.LayerNorm()
        self.attn = nn.SelfAttention(num_heads=4)
        self.norm2 = nn.LayerNorm()
        self.ffn = nn.Sequential([
            nn.Dense(self.dim * 4),
            nn.silu,
            nn.Dense(self.dim)
        ])
    
    def __call__(self, x):
        b, n, d = x.shape
        # Pad
        if n % self.segment_len != 0:
            pad_len = self.segment_len - (n % self.segment_len)
            x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))
            n = x.shape[1]
            
        num_segments = n // self.segment_len
        x_reshaped = x.reshape(b, num_segments, self.segment_len, d)
        
        # Helper for loop over segments
        # We use scan again
        
        def segment_fn(memory_state, seg):
            # seg: [b, seg_len, d]
            
            # 1. Retrieve
            h_t = self.neural_memory.retrieve(seg, memory_state)
            
            # 2. Persistent
            p_mem = self.persistent_memory(b)
            
            # 3. Concat
            # Context = [P, h, S]
            context = jnp.concatenate([p_mem, h_t, seg], axis=1)
            
            # 4. Attention
            # Causal mask?
            # Flax SelfAttention handles masking if we pass mask.
            # Make causal mask for the whole context
            L = context.shape[1]
            mask = nn.make_causal_mask(jnp.ones((b, L)), dtype=jnp.bool)
            
            attn_out = self.attn(context, mask)
            
            # Take segment part (last segment_len)
            y_t = attn_out[:, -self.segment_len:, :]
            y_t = self.norm1(y_t + seg)
            
            # FFN
            y_t = y_t + self.ffn(self.norm2(y_t))
            
            # 5. Update Memory
            _, new_memory_state = self.neural_memory(y_t, memory_state)
            
            return new_memory_state, y_t
            
        _, final_outputs_stacked = lax.scan(segment_fn, None, jnp.swapaxes(x_reshaped, 0, 1))
        
        # Stacked outputs: [num_segments, b, seg_len, d]
        # Transpose to [b, num_segments, seg_len, d]
        final_outputs = jnp.swapaxes(final_outputs_stacked, 0, 1)
        final_outputs = final_outputs.reshape(b, -1, d)
        
        # Crop padding
        return final_outputs[:, :x.shape[1]- (self.segment_len - (n % self.segment_len)) if n%self.segment_len!=0 else None, :]

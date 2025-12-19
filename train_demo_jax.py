
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from titans_jax import TitansMAC
import jax.random as random

def train_dummy_jax():
    dim = 32
    memory_dim = 32
    segment_len = 8
    seq_len = 32
    batch_size = 4
    lr = 0.001
    steps = 50
    
    # Initialize Key
    key = random.PRNGKey(0)
    key, init_key, data_key = random.split(key, 3)
    
    # Initialize Model
    model = TitansMAC(dim=dim, memory_dim=memory_dim, segment_len=segment_len)
    
    # Dummy input for init
    dummy_input = jnp.ones((batch_size, seq_len, dim))
    params = model.init(init_key, dummy_input)
    
    # Optimizer
    tx = optax.adamw(learning_rate=lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    # Step function
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            logits = state.apply_fn(params, batch)
            # MSE Loss against itself (Reconstruction/Identity task)
            loss = jnp.mean((logits - batch) ** 2)
            return loss
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    print("Starting JAX Dummy Training Loop...")
    
    for step in range(steps):
        # Generate random data
        data_key, subkey = random.split(data_key)
        batch = random.normal(subkey, (batch_size, seq_len, dim))
        
        state, loss = train_step(state, batch)
        
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss:.4f}")
            
    print("JAX Training loop completed.")

if __name__ == "__main__":
    try:
        train_dummy_jax()
        print("Verification SUCCESS.")
    except Exception as e:
        print(f"Verification FAILED with error: {e}")
        import traceback
        traceback.print_exc()

import torch
import torch.nn as nn
import torch.optim as optim
from titans import TitansMAC

def train_dummy():
    # Hyperparameters
    dim = 32
    memory_dim = 32 # Assuming square memory for now
    segment_len = 8
    seq_len = 32
    batch_size = 4
    lr = 0.001
    steps = 50
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = TitansMAC(dim=dim, memory_dim=memory_dim, segment_len=segment_len).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    print("Starting Dummy Training Loop...")
    model.train()
    
    for step in range(steps):
        # Dummy Data: Predict next vector (simple shift)
        # Input: Random sequence
        x = torch.randn(batch_size, seq_len, dim).to(device)
        
        # Target: Shifted x (predict next step)
        # Just simple regression: Target is x shifted by 1 + noise, or just x * 0.5
        # Let's do reconstruction of x for simplicity to see if it learns identity/mapping.
        target = x 
        
        # Forward
        output = model(x)
        
        # Loss
        loss = loss_fn(output, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item()
                
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}, Grad Norm = {total_norm:.4f}")
            
        optimizer.step()
        
    print("Training loop completed.")

if __name__ == "__main__":
    try:
        train_dummy()
        print("Verification SUCCESS.")
    except Exception as e:
        print(f"Verification FAILED with error: {e}")
        import traceback
        traceback.print_exc()

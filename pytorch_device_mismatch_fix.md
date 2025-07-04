# PyTorch Device Mismatch Fix for DexterousHand Project

## Error Analysis

The error occurs in `/home/smy/dexterousHand/mbrl/controller/controller.py` at line 105:

```python
over_threshold = (value > c)  # shape [nopt, npart]
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

This indicates that `value` and `c` tensors are on different devices (one on CUDA, one on CPU).

## Root Cause

The issue typically happens when:
1. Model parameters/tensors are moved to GPU but input data remains on CPU
2. Constants or thresholds are created on CPU while computation tensors are on GPU
3. Inconsistent device management across different parts of the code

## Solution

### 1. Immediate Fix for Line 105

In `controller.py` around line 105, ensure both tensors are on the same device:

```python
# Before the comparison, ensure both tensors are on the same device
if hasattr(value, 'device'):
    c = c.to(value.device)
elif hasattr(c, 'device'):
    value = value.to(c.device)

over_threshold = (value > c)  # shape [nopt, npart]
```

### 2. Better Solution - Consistent Device Management

Add device consistency throughout the `mpc_cost_fun` method:

```python
def mpc_cost_fun(self, ac_seqs, states, Qfactor, return_torch, **kwargs):
    # Determine the target device from the input tensors
    if hasattr(states, 'device'):
        device = states.device
    elif hasattr(ac_seqs, 'device'):
        device = ac_seqs.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure all inputs are on the same device
    if hasattr(ac_seqs, 'to'):
        ac_seqs = ac_seqs.to(device)
    if hasattr(states, 'to'):
        states = states.to(device)
    if hasattr(Qfactor, 'to'):
        Qfactor = Qfactor.to(device)
    
    # ... rest of the method ...
    
    # Before line 105, ensure 'c' is on the same device as 'value'
    if isinstance(c, torch.Tensor):
        c = c.to(device)
    else:
        c = torch.tensor(c, device=device)
    
    over_threshold = (value > c)  # shape [nopt, npart]
```

### 3. Systematic Fix - Device Management Class

Create a device manager utility:

```python
import torch

class DeviceManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def to_device(self, tensor_or_value):
        """Move tensor or convert value to the target device"""
        if isinstance(tensor_or_value, torch.Tensor):
            return tensor_or_value.to(self.device)
        else:
            return torch.tensor(tensor_or_value, device=self.device)
    
    def ensure_same_device(self, *tensors):
        """Ensure all tensors are on the same device"""
        if not tensors:
            return tensors
        
        # Use the device of the first tensor as reference
        target_device = tensors[0].device if hasattr(tensors[0], 'device') else self.device
        
        result = []
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                result.append(tensor.to(target_device))
            else:
                result.append(torch.tensor(tensor, device=target_device))
        
        return result if len(result) > 1 else result[0]
```

### 4. Integration in Controller Class

Modify the controller class to use consistent device management:

```python
class Controller:
    def __init__(self, *args, **kwargs):
        # ... existing initialization ...
        self.device_manager = DeviceManager()
    
    def mpc_cost_fun(self, ac_seqs, states, Qfactor, return_torch, **kwargs):
        # Ensure all inputs are on the same device
        ac_seqs, states, Qfactor = self.device_manager.ensure_same_device(ac_seqs, states, Qfactor)
        
        # ... rest of the method ...
        
        # Before the problematic line
        value, c = self.device_manager.ensure_same_device(value, c)
        over_threshold = (value > c)  # shape [nopt, npart]
```

## Files to Check and Modify

1. **`/home/smy/dexterousHand/mbrl/controller/controller.py`** (Line 105)
2. **`/home/smy/dexterousHand/mbrl/controller/MPC.py`** (Check device consistency in `sample` method)
3. **`/home/smy/dexterousHand/mbrl/controller/CEM.py`** (Check device consistency in `obtain_solution`)
4. **`/home/smy/dexterousHand/mbrl/agent/dpetsAlpha.py`** (Check device consistency in `sample`)

## Quick Debug Commands

To debug the specific tensors causing the issue:

```python
# Add these debug prints before line 105 in controller.py
print(f"value device: {value.device if hasattr(value, 'device') else 'not a tensor'}")
print(f"c device: {c.device if hasattr(c, 'device') else 'not a tensor'}")
print(f"value type: {type(value)}")
print(f"c type: {type(c)}")
```

## Prevention

1. Always check tensor devices before operations
2. Use a consistent device management strategy throughout the codebase
3. Move all model parameters and data to the same device during initialization
4. Use `tensor.to(device)` consistently when creating new tensors

## Testing

After applying the fix, test with:
1. CPU-only mode (set CUDA_VISIBLE_DEVICES="")
2. GPU mode with CUDA
3. Mixed precision if used in the codebase

This should resolve the device mismatch error and prevent similar issues in the future.
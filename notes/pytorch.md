# Pytorch notes

1. All classes which inherit from nn.Module run the forward method when called. 


Example:

```python
class One(nn.Module):
    def forward(self, txt: str):
        print(txt)
# ---
instance1 = One()
instance1("Hello") 
# Output: Hello
```

2. torch.arange(X, Y) -> tensor([X, ..., Y-1])
```python
torch.arange(1, 8) = torch([1,2,3,4,5,6,7])
```
3. .expand()
The expand function in PyTorch is used to broadcast a tensor to a larger size without actually copying the data in memory. It's a memory-efficient way to repeat tensor dimensions.

    torch.arange(0, seq_length) creates a 1D tensor: [0, 1, 2, ..., seq_length-1]

    .expand(N, seq_length) broadcasts this 1D tensor to shape (N, seq_length)
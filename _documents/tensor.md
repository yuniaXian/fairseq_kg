[link] https://stackoverflow.com/questions/56088189/pytorch-how-can-i-find-indices-of-first-nonzero-element-in-each-row-of-a-2d-ten
```
idx = torch.arange(tmp.shape[1], 0, -1)
tmp2= tmp * idx
indices = torch.argmax(tmp2, 1, keepdim=True)

```
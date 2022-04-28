<!-- Chuan hóa -->

```python
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
normalizer = normalizers.Sequence([NFD(), StripAccents()])
normalizer.normalize_str("Héllò hôw are ü?")
```
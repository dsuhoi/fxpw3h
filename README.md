# fxpw3h
Модель, реализующая прогнозы по одночасовой погодной сетке с использованием каскада моделей FuXi и Pangu-Weather. Пример запуска:

```python
from fxpw3h import FuXiPW
import xarray as xr
from pathlib import Path

inp = xr.load_dataarray("input_data.nc")

model = FuXiPW(Path("models_weights_dir/"))

res = model.run_complete(inp, [21, 0, 0]) # запуск на 120 часов прогнозирования

res.to_netcdf("output.nc")
```

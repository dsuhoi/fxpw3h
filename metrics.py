import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm


def weighted_rmse(
    out: np.ndarray, target: xr.DataArray, coord_names: tuple[str] = ("lat", "lon")
) -> xr.DataArray:
    """Взвешенное RMSE по широте."""
    wlat = np.cos(np.deg2rad(target[coord_names[0]]))
    wlat /= wlat.mean()
    error = (out - target) ** 2 * wlat
    return np.sqrt(error.mean(*coord_names))


def animate_imshow(
    data: np.ndarray, frames: int | None = None, fname: str | None = None
) -> None:
    """Функция визуализации результатов работы модели (получение GIF анимации прогнозов)"""
    if frames is None:
        frames = data.shape[0]
    fig, ax = plt.subplots()
    pbar = tqdm(total=frames)

    def animate(i):
        pbar.update(1)
        ax.imshow(data[i])

    ani = anim.FuncAnimation(fig, animate, frames=frames)
    if fname:
        ani.save(fname)
    else:
        plt.show()

    pbar.close()

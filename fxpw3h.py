import pathlib
import time
from datetime import datetime

import numpy as np
import onnxruntime as ort
import pandas as pd
import scipy as sp
import xarray as xr

DEFAULT_FUXI_PATH = pathlib.Path("weights/").resolve()


class FuXi:
    """Реализация модели FuXi."""

    def __init__(self, model_dir: pathlib.Path = DEFAULT_FUXI_PATH):
        self.model_dir = model_dir

    @staticmethod
    def load_model(model_name: str) -> ort.InferenceSession:
        """Загрузка весов модели."""
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.intra_op_num_threads = 1

        session = ort.InferenceSession(
            model_name, sess_options=options, providers=["CUDAExecutionProvider"]
        )
        return session

    @staticmethod
    def time_encoding(
        init_time: datetime, total_step: int, freq: int = 6
    ) -> np.ndarray:
        """Кодирование временной сетки для запуска модели FuXi."""
        init_time = np.array([init_time])
        tembs = []
        for i in range(total_step):
            hours = np.array([pd.Timedelta(hours=t * freq) for t in [i - 1, i, i + 1]])
            times = init_time[:, None] + hours[None]
            times = [pd.Period(t, "h") for t in times.reshape(-1)]
            times = [(p.day_of_year / 366, p.hour / 24) for p in times]
            temb = np.array(times, dtype=np.float32)
            temb = np.concatenate([np.sin(temb), np.cos(temb)], axis=-1)
            temb = temb.reshape(1, -1)
            tembs.append(temb)
        return np.stack(tembs)

    def run_complete(
        self, init_data: xr.DataArray, num_steps: list[int]
    ) -> xr.DataArray:
        """Запуск 6-часовой модели FuXi на num_steps итерациях для трех подмоделей."""
        lat, lon = init_data.lat.values, init_data.lon.values
        curr_time = init_data.time[1].values + pd.Timedelta(hours=6)
        level = init_data.level.values

        total_step = sum(num_steps)
        init_time = pd.to_datetime(init_data.time.values[-1])
        tembs = FuXi.time_encoding(init_time, total_step)

        print(f'init_time: {init_time.strftime(("%Y%m%d-%H"))}')
        print(f"latitude: {init_data.lat.values[0]} ~ {init_data.lat.values[-1]}")

        assert init_data.lat.values[0] == 90
        assert init_data.lat.values[-1] == -90

        input = init_data.values[None]
        print(np.array(input).shape)
        print(f"input: {input.shape}, {input.min():.2f} ~ {input.max():.2f}")
        print(f"tembs: {tembs.shape}, {tembs.mean():.4f}")

        stages = ["short", "medium", "long"]
        step = 0
        res = []
        for i, num_step in enumerate(num_steps):
            if num_step <= 0:
                continue
            stage = stages[i]
            start = time.perf_counter()
            model_name = self.model_dir / f"{stage}.onnx"
            print(f"Load model from {model_name} ...")

            session = FuXi.load_model(str(model_name))
            load_time = time.perf_counter() - start
            print(f"Load model take {load_time:.2f} sec")

            print(f"Inference {stage} ...")
            start = time.perf_counter()

            for _ in range(0, num_step):
                temb = tembs[step]
                (new_input,) = session.run(
                    None, {"input": np.array(input), "temb": temb}
                )
                output = new_input[:, -1]

                res.append(
                    xr.DataArray(
                        data=output.copy(),
                        dims=["time", "level", "lat", "lon"],
                        coords={
                            "time": [curr_time],
                            "level": level,
                            "lat": lat,
                            "lon": lon,
                        },
                    )
                )
                del input
                print(f"stage: {i}, step: {step + 1:02d}")
                input = new_input
                step += 1
                curr_time += pd.Timedelta(hours=6)

            run_time = time.perf_counter() - start
            print(f"Inference {stage} take {run_time:.2f}")

            del session

            if step > total_step:
                break
        return xr.concat(res, dim="time")


class FuXiPW(FuXi):
    def __init__(self, model_dir: pathlib.Path = DEFAULT_FUXI_PATH):
        super().__init__(model_dir)

    @staticmethod
    def fuxi2pangu(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Преобразование данных из FuXi в PanguWeather формат."""
        z = data[12::-1]
        t = data[25:12:-1]
        u = data[38:25:-1]
        v = data[51:38:-1]
        q = data[64:51:-1] / 2e5
        msl = data[-2]
        u10 = data[-4]
        v10 = data[-3]
        t2m = data[-5]
        inpn = np.stack([z, q, t, u, v]).astype(np.float32)
        inp_surfn = np.stack([msl, u10, v10, t2m]).astype(np.float32)
        return inp_surfn, inpn

    @staticmethod
    def interp_1d(data, h=6, method="linear") -> np.ndarray:
        """Интерполяция по часовой сетке."""
        n_old = np.linspace(0, 1, data.shape[0])
        N = np.linspace(0, 1, data.shape[0] * h - h + 1)
        res = sp.interpolate.interpn([n_old], data, N, method=method)
        return res

    def run_complete(
        self, init_data: xr.DataArray, num_steps: list[int]
    ) -> xr.DataArray:
        """Запуск комплексной модели."""
        res = super().run_complete(init_data, num_steps)
        print("FuXi END!")
        lat, lon = res.lat.values, init_data.lon.values
        curr_time = res.time[0].values

        interp_tp = FuXiPW.interp_1d(res[:, -1].values)

        pw_model = super().load_model(self.model_dir / "pangu_weather_3.onnx")

        outs = []
        for i, r in enumerate(res[:-1]):  # на каждые 6 часов по 5 прогнозов
            inp_s, inp_up = FuXiPW.fuxi2pangu(r.values)
            outs.append(inp_s)
            output, output_surface = pw_model.run(
                None, {"input": inp_up, "input_surface": inp_s}
            )
            outs.append(output_surface.copy())
            print(f"Часы от начала прогноза {6 * i} h.\nИтерация {i + 1} завершена!")

        del pw_model

        outs = np.array(outs + [FuXiPW.fuxi2pangu(res[-1].values)[0]])
        res = FuXiPW.interp_1d(outs, h=3)
        res = np.append(res, interp_tp.reshape((-1, 1, 721, 1440)), axis=1)

        return xr.DataArray(
            data=res,
            dims=["time", "level", "lat", "lon"],
            coords={
                "time": [
                    curr_time + i * pd.Timedelta(hours=1) for i in range(res.shape[0])
                ],
                "level": ["MSLP", "U10", "V10", "T2M", "TP"],
                "lat": lat,
                "lon": lon,
            },
        ).astype(np.float32)

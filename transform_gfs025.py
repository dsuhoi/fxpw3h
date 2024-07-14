import os
import pathlib
import sys
from datetime import datetime, timedelta

import cfgrib
import numpy as np
import pandas as pd
import pygrib as pg
import xarray as xr


def get_tp(src_name: str):
    """Получение количества осадков из GRIB файла."""
    ds = pg.open(src_name)
    img, _, _ = ds.select(shortName="tp")[0].data()
    return img


def _gfs2nc(
    src_name: str, step: int, prev_tp_img: np.ndarray | None = None
) -> xr.DataArray:
    """Функция конвертации данных из GRIB в Xarray DataArray."""
    assert os.path.exists(src_name)
    print(src_name)
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    pl_names = ["gh", "t", "u", "v", "r"]  # if pl_flag else []
    sf_names = ["2t", "10u", "10v", "mslet"]

    try:
        ds = pg.open(src_name)
    except Exception():
        print(f"{src_name} not found")
        return

    input = []
    level = []

    for name in pl_names + sf_names + ["tp"]:

        if name in pl_names:
            try:
                data = ds.select(shortName=name, level=levels)
            except Exception():
                print("pl wrong")
                return

            data = data[: len(levels)]

            if len(data) != len(levels):
                print("pl wrong")
                return

            if name == "gh":
                name = "z"

            for v in data:
                init_time = f"{v.date}-{v.time // 100:02d}"

                lat = v.distinctLatitudes
                lon = v.distinctLongitudes

                img, _, _ = v.data()

                if name == "z":
                    img = img * 9.8

                input.append(img)
                level.append(f"{name}{v.level}")
                print(f"{v.name}: {v.level}, {img.shape}, {img.min()} ~ {img.max()}")

        if name in sf_names:
            try:
                data = ds.select(shortName=name)
            except Exception():
                print("sfc wrong")
                return

            name_map = {"2t": "t2m", "10u": "u10", "10v": "v10", "mslet": "msl"}
            name = name_map[name]

            for v in data:
                img, _, _ = v.data()
                input.append(img)
                level.append(name)
                print(f"{v.name}: {img.shape}, {img.min()} ~ {img.max()}")

        if name == "tp":
            if step % 6 != 0:
                tp, _, _ = ds.select(shortName="tp")[0].data()
                tp -= prev_tp_img
            else:
                tp = prev_tp_img
            input.append(tp / 1000)
            level.append("tp")
            print(f"Total Precipitation: {tp.shape}, {tp.min()} ~ {tp.max()}")

    input = np.stack(input)

    init_time = f"{data[0].date}-{step:02d}"
    lat = data[0].distinctLatitudes
    lon = data[0].distinctLongitudes

    times = [pd.to_datetime(init_time)]
    input = xr.DataArray(
        data=input[None],
        dims=["time", "level", "lat", "lon"],
        coords={"time": times, "level": level, "lat": lat, "lon": lon},
    )

    if np.isnan(input).sum() > 0:
        print("Field has nan value")
        return

    return input


def parsing_files(target_dir: str, output_dir: str):
    """Обработка данных с учетом особенностей структуры хранения GFS025 снимков."""
    paths = [pathlib.Path(i) for i in target_dir]
    # функция поиска предыдущего дня
    prev_day = lambda d: datetime.strftime(
        (datetime.strptime(d, "%Y%m%d") - timedelta(hours=1)), "%Y%m%d"
    )
    day_type = ["00", "06", "12", "18"]  # прогнозы по 6-часовым циклам
    for p in paths:
        print(p)
        df_list = []
        tp = None  # значение кол-ва осадков за прошлый час
        for i in range(24):
            dts = day_type[i // 6]
            file = p / dts / f"gfs.t{dts}z.pgrb2.0p25.f00{i % 6}"
            if i % 6 == 0:  # условия на обработку данных об осадках
                if i == 0:
                    src_name = p.parent / prev_day(p.stem)
                    tp_dts = "18"
                else:
                    src_name = p
                    tp_dts = day_type[(i // 6) - 1]
                path = src_name / tp_dts / f"gfs.t{tp_dts}z.pgrb2.0p25.f006"
                tp = get_tp(str(path)) - get_tp(f"{str(path)[:-1]}5")
            elif i % 6 == 1:
                tp = np.zeros_like(tp)
            elif i % 6 > 1:
                tp = get_tp(str(p / dts / f"gfs.t{dts}z.pgrb2.0p25.f00{i % 6 - 1}"))
            if not (res := _gfs2nc(str(file), i, tp)) is None:
                df_list.append(res)
        df = xr.concat(df_list, "time")
        del df_list
        df.astype(np.float32).to_netcdf(str(pathlib.Path(output_dir) / f"{p.stem}.nc"))
        del df


def find_folders(month: str, base_dir: str, output_dir: str, year: str):
    """Поиск всех директорий с днями прогнозов"""
    days = list(pathlib.Path(base_dir).glob(f"{year}{month}*"))
    parsing_files(days, output_dir)


def start_process():
    """Функция запуска скрипта. CLI параметры:
    - месяц выгрузки;
    - путь до директории с файлами результата;
    - путь до директории с файлами выгрузки;"""
    month = str(sys.argv[1])
    output_path = sys.argv[2]
    base_dir = sys.argv[3]

    year = "2022"
    if len(sys.argv) > 3:
        year = sys.argv[4]
    find_folders(month, base_dir, output_path, year)


if __name__ == "__main__":
    start_process()

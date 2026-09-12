import os
import re
import json
import math
import pickle
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

try:
    from borax.calendars import LunarDate
except Exception:
    LunarDate = None


# ============================================================
# basic utils
# ============================================================

def expand_path(p):
    return str(Path(p).expanduser().resolve())


def normalize_region_str(x):
    s = str(x)
    nums = re.findall(r"-?\d+", s)
    if len(nums) < 2:
        return s
    return f"[{int(nums[0])}, {int(nums[1])}]"


def parse_any_datetime(x):
    if pd.isna(x):
        return pd.NaT

    s = str(x)

    try:
        t = pd.to_datetime(s)
        if not pd.isna(t):
            return t
    except Exception:
        pass

    nums = re.sub(r"[^0-9]+", "", s)

    if len(nums) >= 14:
        return pd.Timestamp(datetime.strptime(nums[:14], "%Y%m%d%H%M%S"))
    if len(nums) >= 8:
        return pd.Timestamp(datetime.strptime(nums[:8], "%Y%m%d"))

    raise ValueError(f"Cannot parse datetime: {x}")


def get_mag_col(df):
    for c in ["magnitude", "mag", "MAG", "Magnitude"]:
        if c in df.columns:
            return c
    raise ValueError("Catalog must contain magnitude / mag column.")


def get_depth_col(df):
    for c in ["depth", "DEPTH", "Depth"]:
        if c in df.columns:
            return c
    raise ValueError("Catalog must contain depth column.")


def get_lon_lat_cols(df):
    lon_col = None
    lat_col = None

    for c in ["lon", "longitude", "LONGITUDE", "Long", "LON"]:
        if c in df.columns:
            lon_col = c
            break

    for c in ["lat", "latitude", "LATITUDE", "Lat", "LAT"]:
        if c in df.columns:
            lat_col = c
            break

    return lon_col, lat_col


def date_range_daily(start_date, end_date, drop_feb29=True):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    dates = []
    cur = start.normalize()

    while cur <= end:
        if drop_feb29 and cur.month == 2 and cur.day == 29:
            pass
        else:
            dates.append(cur)
        cur += pd.Timedelta(days=1)

    return dates


def get_level_array(x, thresholds, withequal=True):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, -1, dtype=int)

    for i, th in enumerate(thresholds):
        if withequal:
            out[x >= th] = i
        else:
            out[x > th] = i

    return out


# ============================================================
# patch and catalog loading
# ============================================================

def load_patch_table(patch_list_path):
    patch = pd.read_csv(expand_path(patch_list_path))

    if "need" in patch.columns:
        patch = patch[patch["need"] == True].copy()

    if "xy" not in patch.columns:
        raise ValueError("patch list must contain column 'xy'.")

    patch = patch.reset_index(drop=True)
    patch["region"] = patch["xy"].astype(str).map(normalize_region_str)
    patch["patch_order"] = np.arange(len(patch), dtype=int)

    return patch


def assign_region_if_needed(catalog, patch):
    if "region" in catalog.columns:
        catalog["region"] = catalog["region"].map(normalize_region_str)
        return catalog

    required = ["llog", "rlog", "dlat", "ulat"]
    if not all(c in patch.columns for c in required):
        raise ValueError(
            "Catalog has no region column, and patch list lacks llog/rlog/dlat/ulat."
        )

    lon_col, lat_col = get_lon_lat_cols(catalog)
    if lon_col is None or lat_col is None:
        raise ValueError("Catalog has no region column and lacks lon/lat columns.")

    def find_region(row):
        lon = row[lon_col]
        lat = row[lat_col]

        hit = patch[
            (patch["llog"] <= lon)
            & (lon <= patch["rlog"])
            & (patch["dlat"] <= lat)
            & (lat <= patch["ulat"])
        ]

        if len(hit) == 0:
            return None

        return hit.iloc[0]["region"]

    catalog["region"] = catalog.apply(find_region, axis=1)

    return catalog


def load_catalog(catalog_path, patch):
    catalog = pd.read_csv(expand_path(catalog_path))

    if "date_time" in catalog.columns:
        catalog["date_time"] = catalog["date_time"].apply(parse_any_datetime)
    elif "timestep" in catalog.columns:
        catalog["date_time"] = pd.to_datetime(catalog["timestep"], unit="s")
    elif "time" in catalog.columns:
        catalog["date_time"] = catalog["time"].apply(parse_any_datetime)
    else:
        raise ValueError("Catalog must contain date_time / timestep / time.")

    mag_col = get_mag_col(catalog)
    depth_col = get_depth_col(catalog)

    catalog = catalog.rename(columns={mag_col: "magnitude", depth_col: "depth"})
    catalog = assign_region_if_needed(catalog, patch)

    catalog = catalog.dropna(subset=["date_time", "magnitude", "depth", "region"])
    catalog["region"] = catalog["region"].map(normalize_region_str)

    valid_regions = set(patch["region"].tolist())
    catalog = catalog[catalog["region"].isin(valid_regions)].copy()
    catalog = catalog.sort_values("date_time").reset_index(drop=True)

    return catalog


def build_patch_event_arrays(catalog, patch):
    event_dict = {}

    grouped = {
        reg: g.sort_values("date_time")
        for reg, g in catalog.groupby("region")
    }

    for _, row in patch.iterrows():
        patch_order = int(row["patch_order"])
        region = row["region"]

        g = grouped.get(region, pd.DataFrame(columns=["date_time", "magnitude", "depth"]))

        event_dict[patch_order] = {
            "region": region,
            "times": g["date_time"].values.astype("datetime64[ns]"),
            "mags": g["magnitude"].astype(float).values,
            "depths": g["depth"].astype(float).values,
        }

    return event_dict


# ============================================================
# 282-d feature extraction
# ============================================================

def lunar_phase_value(ts):
    if LunarDate is None:
        return 0.0

    ts = pd.Timestamp(ts)
    lun_date = LunarDate.from_solar_date(ts.year, ts.month, ts.day)

    parts = str(lun_date).split("(")[1].split(")")[0].split(",")
    lun_year = int(parts[0])
    lun_month = int(parts[1])
    lun_day = int(parts[2])

    day_num = 30
    try:
        LunarDate(lun_year, lun_month, 30)
    except ValueError:
        day_num = 29

    phase = lun_day / day_num * 2 * math.pi
    return math.sin(phase) + 2


def get_ab_value(mags, need_mse=False):
    mags = np.asarray(mags, dtype=float)

    if len(mags) == 0:
        if need_mse:
            return 0.0, 0.0, 0.0
        return 0.0, 0.0

    values, counts = np.unique(mags, return_counts=True)
    order = np.argsort(values)

    values = values[order]
    counts = counts[order]

    if len(values) <= 3:
        if need_mse:
            return 0.0, 0.0, 0.0
        return 0.0, 0.0

    start = int(np.argmax(counts))

    x = values[start:-1].reshape(-1, 1)
    y = np.log(counts[start:-1])

    if len(x) > 2:
        lr = LinearRegression()
        lr.fit(x, y)

        b_value = float(lr.coef_[0])
        a_value = float(lr.intercept_)

        if need_mse:
            pred = lr.predict(x)
            mse = float(mean_squared_error(y, pred))
            return b_value, a_value, mse

        return b_value, a_value

    if need_mse:
        return 0.0, 0.0, 0.0

    return 0.0, 0.0


def top5_and_lunar(times, mags):
    if len(mags) == 0:
        return [0.0] * 5, [0.0] * 5

    time_int = times.astype("datetime64[ns]").astype(np.int64)
    order = np.lexsort((time_int, -mags))
    order = order[:5]

    top_mags = list(mags[order].astype(float))
    top_lunar = [lunar_phase_value(pd.Timestamp(times[i])) for i in order]

    if len(top_mags) < 5:
        top_mags += [0.0] * (5 - len(top_mags))
        top_lunar += [0.0] * (5 - len(top_lunar))

    return top_mags, top_lunar


def count_by_levels(values, thresholds):
    levels = get_level_array(values, thresholds, withequal=True)
    out = []

    for i in range(len(thresholds)):
        out.append(int(np.sum(levels == i)))

    return out


def mean_interevent_stats(times, mags, mag_thresholds):
    if len(times) == 0:
        return [0.0] * len(mag_thresholds), [0.0] * len(mag_thresholds)

    levels = get_level_array(mags, mag_thresholds, withequal=True)

    mean_delta = []
    cv_delta = []

    for i in range(len(mag_thresholds)):
        t = times[levels == i]

        if len(t) > 2:
            t = np.sort(t.astype("datetime64[ns]"))
            dt_days = np.diff(t).astype("timedelta64[s]").astype(float) / 86400.0
            m = float(np.mean(dt_days))
            s = float(np.std(dt_days))

            mean_delta.append(m)
            cv_delta.append(s / m if m > 0 else 0.0)
        else:
            mean_delta.append(0.0)
            cv_delta.append(0.0)

    return mean_delta, cv_delta


def compute_282_features_for_patch(event_obj, t0, feature_window_days=365):
    """
    raw 282-d feature definition:

    X(p, t0) uses only events before t0:
        [t0 - feature_window_days, t0)

    Monthly features are relative lag months:
        lag1  = [t0 - 1 month,  t0)
        lag2  = [t0 - 2 months, t0 - 1 month)
        ...
        lag12 = [t0 - 12 months, t0 - 11 months)
    """
    times = event_obj["times"]
    mags = event_obj["mags"]
    depths = event_obj["depths"]

    t0 = pd.Timestamp(t0)
    t0_np = np.datetime64(t0)

    win_start = np.datetime64(t0 - pd.Timedelta(days=feature_window_days))

    left = np.searchsorted(times, win_start, side="left")
    right = np.searchsorted(times, t0_np, side="left")

    win_times = times[left:right]
    win_mags = mags[left:right]
    win_depths = depths[left:right]

    mag_thresholds = [0, 3, 5, 7]
    depth_thresholds = [0, 70]

    # annual features
    annual_top5, annual_lunar5 = top5_and_lunar(win_times, win_mags)

    if len(win_mags) > 0:
        annual_mean = float(np.mean(win_mags))
        annual_std = float(np.std(win_mags))
        annual_b, annual_a = get_ab_value(win_mags)
        annual_prob_m6 = float(math.pow(10, -3 * annual_b)) if annual_b != 0 else 0.0
    else:
        annual_b = 0.0
        annual_a = 0.0
        annual_mean = 0.0
        annual_std = 0.0
        annual_prob_m6 = 0.0

    # relative monthly features
    monthly_b = []
    monthly_a = []
    monthly_mean = []
    monthly_std = []
    monthly_prob_m6 = []
    monthly_top5 = []
    monthly_lunar5 = []
    monthly_mag_counts = []
    monthly_depth_counts = []

    for lag in range(1, 13):
        m_start = np.datetime64(t0 - relativedelta(months=lag))
        m_end = np.datetime64(t0 - relativedelta(months=lag - 1))

        ml = np.searchsorted(times, m_start, side="left")
        mr = np.searchsorted(times, m_end, side="left")

        mtimes = times[ml:mr]
        mmags = mags[ml:mr]
        mdepths = depths[ml:mr]

        if len(mmags) > 0:
            b, a = get_ab_value(mmags)
            monthly_b.append(float(b))
            monthly_a.append(float(a))
            monthly_mean.append(float(np.mean(mmags)))
            monthly_std.append(float(np.std(mmags)))
            monthly_prob_m6.append(float(math.pow(10, -3 * b)) if b != 0 else 0.0)
        else:
            monthly_b.append(0.0)
            monthly_a.append(0.0)
            monthly_mean.append(0.0)
            monthly_std.append(0.0)
            monthly_prob_m6.append(0.0)

        top5, lunar5 = top5_and_lunar(mtimes, mmags)
        monthly_top5.extend(top5)
        monthly_lunar5.extend(lunar5)

        monthly_mag_counts.extend(count_by_levels(mmags, mag_thresholds))
        monthly_depth_counts.extend(count_by_levels(mdepths, depth_thresholds))

    # last 100 events before t0
    hr = np.searchsorted(times, t0_np, side="left")
    hl = max(0, hr - 100)

    last_times = times[hl:hr]
    last_mags = mags[hl:hr]

    if len(last_mags) > 0:
        b100, a100, mse100 = get_ab_value(last_mags, need_mse=True)
        adivb = float(a100 / b100) if b100 != 0 else 0.0
        mean_mag_100 = float(np.mean(last_mags))

        if len(last_mags) == 100:
            span_days = float(
                (last_times[-1] - last_times[0]).astype("timedelta64[s]").astype(float) / 86400.0
            )
            if span_days > 0:
                energy_rate = float(np.sum((10 ** (11.8 + 1.5 * last_mags)) ** 0.5) / span_days)
            else:
                energy_rate = 0.0
        else:
            span_days = 0.0
            energy_rate = 0.0

        mean_delta, sigma_delta = mean_interevent_stats(last_times, last_mags, mag_thresholds)
    else:
        b100 = 0.0
        a100 = 0.0
        adivb = 0.0
        mse100 = 0.0
        mean_mag_100 = 0.0
        span_days = 0.0
        energy_rate = 0.0
        mean_delta = [0.0] * len(mag_thresholds)
        sigma_delta = [0.0] * len(mag_thresholds)

    last100_list = [
        b100,
        a100,
        adivb,
        mse100,
        mean_mag_100,
        span_days,
        energy_rate,
    ] + mean_delta + sigma_delta

    tongji_list = [
        annual_b,
        annual_a,
        annual_mean,
        annual_std,
        annual_prob_m6,
    ]

    tongji_list += monthly_b
    tongji_list += monthly_a
    tongji_list += monthly_mean
    tongji_list += monthly_std
    tongji_list += monthly_prob_m6

    tongji_list += monthly_top5
    tongji_list += monthly_lunar5
    tongji_list += monthly_mag_counts
    tongji_list += monthly_depth_counts

    row = annual_top5 + annual_lunar5 + tongji_list + last100_list

    if len(row) != 282:
        raise ValueError(f"Feature dimension error: got {len(row)}, expected 282")

    return np.asarray(row, dtype=np.float32)


# ============================================================
# labels and split
# ============================================================

def future_max_mag(event_obj, t0, horizon_month):
    times = event_obj["times"]
    mags = event_obj["mags"]

    t0 = pd.Timestamp(t0)
    t1 = np.datetime64(t0)
    t2 = np.datetime64(t0 + relativedelta(months=horizon_month))

    left = np.searchsorted(times, t1, side="right")
    right = np.searchsorted(times, t2, side="right")

    if right <= left:
        return 0.0

    return float(np.max(mags[left:right]))


def mag_to_class(max_mag):
    if max_mag < 5:
        return 0
    if max_mag < 6:
        return 1
    if max_mag < 7:
        return 2
    return 3


def get_common_test_end(catalog_max_date, max_horizon_month):
    return pd.Timestamp(catalog_max_date).normalize() - relativedelta(months=max_horizon_month)


def final_split_for_sample(t0, future_end, args, common_test_end):
    t0 = pd.Timestamp(t0)
    future_end = pd.Timestamp(future_end)

    train_label_end = pd.Timestamp(args.train_label_end)
    test_start = pd.Timestamp(args.test_start)

    if future_end <= train_label_end:
        return "train_pool"

    if (
        t0 >= test_start
        and t0 <= common_test_end
        and t0.day == args.anchor_day
    ):
        return "test"

    return "drop"


def default_inner_folds():
    """
    Inner rolling backtest folds before final test.

    Each fold has:
      - inner training: future_end <= train_label_end
      - one-year embargo after train_label_end
      - monthly common validation anchors
    """
    return [
        {
            "fold": 1,
            "train_label_end": "2002-12-31",
            "valid_start": "2004-01-01",
            "valid_end": "2005-12-01",
        },
        {
            "fold": 2,
            "train_label_end": "2004-12-31",
            "valid_start": "2006-01-01",
            "valid_end": "2007-12-01",
        },
        {
            "fold": 3,
            "train_label_end": "2006-12-31",
            "valid_start": "2008-01-01",
            "valid_end": "2009-12-01",
        },
    ]


def inner_fold_role(t0, future_end, fold_cfg, args):
    t0 = pd.Timestamp(t0)
    future_end = pd.Timestamp(future_end)

    train_label_end = pd.Timestamp(fold_cfg["train_label_end"])
    valid_start = pd.Timestamp(fold_cfg["valid_start"])
    valid_end = pd.Timestamp(fold_cfg["valid_end"])

    if future_end <= train_label_end:
        return "inner_train"

    if (
        t0 >= valid_start
        and t0 <= valid_end
        and t0.day == args.anchor_day
    ):
        return "inner_valid"

    return "drop"


# ============================================================
# feature metadata
# ============================================================

def make_feature_metadata(out_dir):
    rows = []
    idx = 0

    def add(name, time_group, signal_group, month_lag=-1, stat_type="", mag_bin="", depth_bin="", rank=-1):
        nonlocal idx
        rows.append({
            "feature_idx": idx,
            "feature_name": name,
            "time_group": time_group,
            "signal_group": signal_group,
            "month_lag": month_lag,
            "stat_type": stat_type,
            "mag_bin": mag_bin,
            "depth_bin": depth_bin,
            "rank": rank,
        })
        idx += 1

    for r in range(1, 6):
        add(f"annual_top{r}_magnitude", "annual", "magnitude_extreme", stat_type="top_magnitude", rank=r)

    for r in range(1, 6):
        add(f"annual_top{r}_lunar_phase", "annual", "lunar_phase", stat_type="lunar_phase", rank=r)

    annual_stats = [
        ("annual_b_value", "magnitude_distribution", "b_value"),
        ("annual_a_value", "magnitude_distribution", "a_value"),
        ("annual_mean_magnitude", "magnitude_distribution", "mean_magnitude"),
        ("annual_std_magnitude", "magnitude_distribution", "std_magnitude"),
        ("annual_prob_density_Mge6", "magnitude_distribution", "prob_density_Mge6"),
    ]

    for name, group, stat in annual_stats:
        add(name, "annual", group, stat_type=stat)

    monthly_blocks = [
        ("monthly_b_value", "magnitude_distribution", "b_value"),
        ("monthly_a_value", "magnitude_distribution", "a_value"),
        ("monthly_mean_magnitude", "magnitude_distribution", "mean_magnitude"),
        ("monthly_std_magnitude", "magnitude_distribution", "std_magnitude"),
        ("monthly_prob_density_Mge6", "magnitude_distribution", "prob_density_Mge6"),
    ]

    for prefix, group, stat in monthly_blocks:
        for lag in range(1, 13):
            add(
                f"{prefix}_lag{lag:02d}",
                "monthly_lag",
                group,
                month_lag=lag,
                stat_type=stat,
            )

    for lag in range(1, 13):
        for r in range(1, 6):
            add(
                f"monthly_lag{lag:02d}_top{r}_magnitude",
                "monthly_lag",
                "magnitude_extreme",
                month_lag=lag,
                stat_type="top_magnitude",
                rank=r,
            )

    for lag in range(1, 13):
        for r in range(1, 6):
            add(
                f"monthly_lag{lag:02d}_top{r}_lunar_phase",
                "monthly_lag",
                "lunar_phase",
                month_lag=lag,
                stat_type="lunar_phase",
                rank=r,
            )

    mag_bins = ["Mge0_lt3", "Mge3_lt5", "Mge5_lt7", "Mge7"]

    for lag in range(1, 13):
        for mb in mag_bins:
            add(
                f"monthly_lag{lag:02d}_count_{mb}",
                "monthly_lag",
                "seismicity_rate",
                month_lag=lag,
                stat_type="event_count",
                mag_bin=mb,
            )

    depth_bins = ["depth_ge0_lt70", "depth_ge70"]

    for lag in range(1, 13):
        for db in depth_bins:
            add(
                f"monthly_lag{lag:02d}_count_{db}",
                "monthly_lag",
                "depth_distribution",
                month_lag=lag,
                stat_type="event_count",
                depth_bin=db,
            )

    last100_base = [
        ("last100_b_value", "magnitude_distribution", "b_value"),
        ("last100_a_value", "magnitude_distribution", "a_value"),
        ("last100_a_div_b", "magnitude_distribution", "a_div_b"),
        ("last100_mse_gr_fit", "magnitude_distribution", "mse_gr_fit"),
        ("last100_mean_magnitude", "magnitude_distribution", "mean_magnitude"),
        ("last100_time_span_days", "temporal_clustering", "time_span_days"),
        ("last100_energy_rate", "energy_release", "energy_rate"),
    ]

    for name, group, stat in last100_base:
        add(name, "last100", group, stat_type=stat)

    for mb in mag_bins:
        add(
            f"last100_mean_delta_day_{mb}",
            "last100",
            "temporal_clustering",
            stat_type="mean_interevent_time",
            mag_bin=mb,
        )

    for mb in mag_bins:
        add(
            f"last100_sigma_delta_day_{mb}",
            "last100",
            "temporal_clustering",
            stat_type="cv_interevent_time",
            mag_bin=mb,
        )

    df = pd.DataFrame(rows)

    if len(df) != 282:
        raise ValueError(f"Feature metadata length error: got {len(df)}")

    path = os.path.join(out_dir, "feature_metadata_282_relative_lag.csv")
    df.to_csv(path, index=False)

    return path


# ============================================================
# main dataset construction
# ============================================================

def build_one_horizon(H, patch, event_dict, feature_dates, common_test_end, args, fold_cfgs):
    X_list = []
    y_class_list = []
    y_m5_list = []
    meta_rows = []

    print("\n" + "=" * 100, flush=True)
    print(f"Building horizon H={H} months", flush=True)

    for _, prow in tqdm(patch.iterrows(), total=len(patch)):
        patch_order = int(prow["patch_order"])
        region = prow["region"]
        ev = event_dict[patch_order]

        for t0 in feature_dates:
            future_end = pd.Timestamp(t0) + relativedelta(months=H)

            split = final_split_for_sample(
                t0=t0,
                future_end=future_end,
                args=args,
                common_test_end=common_test_end,
            )

            if split == "drop":
                continue

            feat = compute_282_features_for_patch(
                ev,
                t0=t0,
                feature_window_days=args.feature_window_days,
            )

            max_mag = future_max_mag(ev, t0, H)
            y_class = mag_to_class(max_mag)
            y_m5 = int(y_class > 0)

            row = {
                "t0": str(pd.Timestamp(t0).date()),
                "split": split,
                "horizon_month": H,
                "patch_order": patch_order,
                "region": region,
                "future_start_exclusive": str(pd.Timestamp(t0).date()),
                "future_end_inclusive": str(pd.Timestamp(future_end).date()),
                "future_max_mag": max_mag,
                "y_class": y_class,
                "y_m5": y_m5,
            }

            for fold_cfg in fold_cfgs:
                f = int(fold_cfg["fold"])
                row[f"fold{f}_role"] = inner_fold_role(
                    t0=t0,
                    future_end=future_end,
                    fold_cfg=fold_cfg,
                    args=args,
                )

            X_list.append(feat)
            y_class_list.append(y_class)
            y_m5_list.append(y_m5)
            meta_rows.append(row)

    X = np.vstack(X_list).astype(np.float32)
    y_class = np.asarray(y_class_list, dtype=np.int64)
    y_m5 = np.asarray(y_m5_list, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)

    return X, y_class, y_m5, meta


def summarize_dataset(meta, y_class, H, fold_cfgs):
    rows = []

    for sp in ["train_pool", "test"]:
        idx = meta["split"].eq(sp).values
        yy = y_class[idx]

        row = {
            "horizon_month": H,
            "subset": sp,
            "n": int(idx.sum()),
            "class0": int(np.sum(yy == 0)),
            "class1": int(np.sum(yy == 1)),
            "class2": int(np.sum(yy == 2)),
            "class3": int(np.sum(yy == 3)),
            "m5_positive": int(np.sum(yy > 0)),
        }

        if idx.sum() > 0:
            row["t0_min"] = str(meta.loc[idx, "t0"].min())
            row["t0_max"] = str(meta.loc[idx, "t0"].max())
            row["future_end_min"] = str(meta.loc[idx, "future_end_inclusive"].min())
            row["future_end_max"] = str(meta.loc[idx, "future_end_inclusive"].max())

        rows.append(row)

    for fold_cfg in fold_cfgs:
        f = int(fold_cfg["fold"])
        col = f"fold{f}_role"

        for role in ["inner_train", "inner_valid"]:
            idx = meta[col].eq(role).values
            yy = y_class[idx]

            row = {
                "horizon_month": H,
                "subset": f"fold{f}_{role}",
                "n": int(idx.sum()),
                "class0": int(np.sum(yy == 0)),
                "class1": int(np.sum(yy == 1)),
                "class2": int(np.sum(yy == 2)),
                "class3": int(np.sum(yy == 3)),
                "m5_positive": int(np.sum(yy > 0)),
            }

            if idx.sum() > 0:
                row["t0_min"] = str(meta.loc[idx, "t0"].min())
                row["t0_max"] = str(meta.loc[idx, "t0"].max())
                row["future_end_min"] = str(meta.loc[idx, "future_end_inclusive"].min())
                row["future_end_max"] = str(meta.loc[idx, "future_end_inclusive"].max())

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--catalog-path",
        default="data/catalog.csv",
    )
    parser.add_argument(
        "--patch-list-path",
        default="data/patch_grid.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw282",
    )

    parser.add_argument("--horizons", default="3,6,12")
    parser.add_argument("--max-horizon-month", type=int, default=12)
    parser.add_argument("--feature-window-days", type=int, default=365)

    parser.add_argument("--train-label-end", default="2010-12-31")
    parser.add_argument("--test-start", default="2012-01-01")
    parser.add_argument("--anchor-day", type=int, default=1)

    parser.add_argument("--feature-start", default=None)
    parser.add_argument("--feature-end", default=None)

    args = parser.parse_args()

    out_dir = expand_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 100, flush=True)
    print("Build raw 282-dim XGBoost datasets with final test and inner rolling folds", flush=True)
    print("catalog:", expand_path(args.catalog_path), flush=True)
    print("patch list:", expand_path(args.patch_list_path), flush=True)
    print("out dir:", out_dir, flush=True)
    print("=" * 100, flush=True)

    if LunarDate is None:
        print("WARNING: borax is not available. Lunar phase features will be set to 0.", flush=True)

    patch = load_patch_table(args.patch_list_path)
    catalog = load_catalog(args.catalog_path, patch)

    catalog_min = catalog["date_time"].min()
    catalog_max = catalog["date_time"].max()

    print("patch number:", len(patch), flush=True)
    print("catalog rows:", len(catalog), flush=True)
    print("catalog time:", catalog_min, "to", catalog_max, flush=True)

    event_dict = build_patch_event_arrays(catalog, patch)

    min_feature_start = (catalog_min + pd.Timedelta(days=args.feature_window_days)).normalize()

    if args.feature_start is None:
        feature_start = min_feature_start
    else:
        feature_start = max(pd.Timestamp(args.feature_start), min_feature_start)

    common_test_end = get_common_test_end(catalog_max, args.max_horizon_month)

    if args.feature_end is None:
        feature_end = common_test_end
    else:
        feature_end = min(pd.Timestamp(args.feature_end), common_test_end)

    feature_dates = date_range_daily(feature_start, feature_end, drop_feb29=True)

    print("feature date range:", feature_dates[0], "to", feature_dates[-1], flush=True)
    print("feature date count:", len(feature_dates), flush=True)
    print("final train_label_end:", args.train_label_end, flush=True)
    print("final test_start:", args.test_start, flush=True)
    print("common_test_end:", common_test_end.date(), flush=True)
    print("anchor_day:", args.anchor_day, flush=True)

    fold_cfgs = default_inner_folds()

    print("\nInner rolling folds:", flush=True)
    for fc in fold_cfgs:
        print(fc, flush=True)

    feature_metadata_path = make_feature_metadata(out_dir)

    patch_info_path = os.path.join(out_dir, "patch_table_used.csv")
    patch.to_csv(patch_info_path, index=False)

    horizons = [int(x) for x in args.horizons.split(",")]

    all_summary = []

    for H in horizons:
        X, y_class, y_m5, meta = build_one_horizon(
            H=H,
            patch=patch,
            event_dict=event_dict,
            feature_dates=feature_dates,
            common_test_end=common_test_end,
            args=args,
            fold_cfgs=fold_cfgs,
        )

        payload = {
            "X": X,
            "y_class": y_class,
            "y_m5": y_m5,
            "meta": meta,
            "horizon_month": H,
            "feature_dim": 282,
            "feature_definition": "[t0 - 365 days, t0), raw unnormalized features",
            "monthly_definition": "relative lag months; lag1=[t0-1m,t0)",
            "label_definition": "(t0, t0 + H months]",
            "final_split_definition": {
                "train_pool": "daily anchors; future_end_inclusive <= train_label_end",
                "embargo": "2011-01-01 to 2011-12-31 by default; not used",
                "test": "monthly common anchors from test_start to catalog_max - max_horizon_month",
            },
            "inner_folds": fold_cfgs,
            "args": vars(args),
        }

        out_pkl = os.path.join(out_dir, f"china_raw282_dataset_horizon_{H}m.pkl")
        out_meta = os.path.join(out_dir, f"china_raw282_dataset_horizon_{H}m_meta.csv")

        with open(out_pkl, "wb") as f:
            pickle.dump(payload, f)

        meta.to_csv(out_meta, index=False)

        summary = summarize_dataset(meta, y_class, H, fold_cfgs)
        summary_path = os.path.join(out_dir, f"summary_{H}m.csv")
        summary.to_csv(summary_path, index=False)

        all_summary.append(summary)

        print("\nSaved:", out_pkl, flush=True)
        print("Saved:", out_meta, flush=True)
        print("X shape:", X.shape, flush=True)
        print(summary.to_string(index=False), flush=True)

    all_summary_df = pd.concat(all_summary, ignore_index=True)
    all_summary_path = os.path.join(out_dir, "summary_all_horizons.csv")
    all_summary_df.to_csv(all_summary_path, index=False)

    manifest = {
        "catalog_path": expand_path(args.catalog_path),
        "patch_list_path": expand_path(args.patch_list_path),
        "out_dir": out_dir,
        "feature_metadata": feature_metadata_path,
        "patch_table_used": patch_info_path,
        "horizons": horizons,
        "max_horizon_month": args.max_horizon_month,
        "feature_definition": "[t0 - 365 days, t0), no future events used",
        "normalization": "none; raw features for XGBoost",
        "final_test": {
            "train_label_end": args.train_label_end,
            "test_start": args.test_start,
            "common_test_end": str(common_test_end.date()),
            "anchor_day": args.anchor_day,
            "description": "final test fixed to monthly common anchors, about 2012-2020 for the current catalog",
        },
        "inner_rolling_backtest_folds": fold_cfgs,
        "args": vars(args),
    }

    manifest_path = os.path.join(out_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nAll done.", flush=True)
    print("summary:", all_summary_path, flush=True)
    print("manifest:", manifest_path, flush=True)


if __name__ == "__main__":
    main()

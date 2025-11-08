import pandas as pd
import yaml
from collections import defaultdict, Counter
from pathlib import Path
import math
import random

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT  = BASE / "output"

def load_data():
    participants = pd.read_csv(DATA / "participants.csv")
    items = pd.read_csv(DATA / "items.csv")
    with open(DATA / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # sanity
    participants["strength"] = participants["strength"].astype(float)
    items["weight_kg"] = items["weight_kg"].astype(float)
    items["day"] = items["day"].astype(str).str.zfill(2)
    items["category"] = items["category"].fillna("regular")
    items["volume"] = items["volume"].str.lower()

    return participants, items, cfg

def choose_spice_carrier(participants, cfg):
    if isinstance(cfg.get("spice_carrier", None), str) and cfg["spice_carrier"].lower() != "auto":
        name = cfg["spice_carrier"]
        if name in set(participants["name"]):
            return name
    # auto: самый сильный (если несколько — первый в таблице)
    idx = participants["strength"].idxmax()
    return participants.loc[idx, "name"]

def compute_targets(participants, items):
    total_w = items["weight_kg"].sum()
    sum_strength = participants["strength"].sum()
    participants = participants.copy()
    participants["target_weight"] = total_w * participants["strength"] / sum_strength
    return participants, total_w

def score_candidate(person, item, state, cfg, spice_carrier):
    """Ниже — простая эвристика (чем меньше — тем лучше)."""
    name = person["name"]
    vol_pen = cfg["volume_penalty"].get(item["volume"], 0.0)

    # 1) Отклонение от цели (после добавления этого предмета)
    new_weight = state["carried_weight"][name] + item["weight_kg"]
    target = person["target_weight"]
    overweight = max(0.0, new_weight - target)
    overweight_cost = cfg["overweight_weight"] * overweight

    # 2) Равномерность по дням: штрафуем за "скопление" одного дня
    days_counter = state["days"][name].copy()
    days_counter[item["day"]] += 1
    # измерим несбалансированность как дисперсию частот
    freqs = list(days_counter.values())
    if freqs:
        mean = sum(freqs) / len(freqs)
        variance = sum((f - mean)**2 for f in freqs) / len(freqs)
    else:
        variance = 0.0
    day_cost = cfg["day_balance_weight"] * variance

    # 3) Объём: небольшой штраф, чтобы большие объёмы не шли слабым
    volume_cost = vol_pen * (1.0 / max(1.0, person["strength"]))

    # 4) Бонус за специи одному человеку
    spice_bonus = 0.0
    if item["category"] == "spice" and name == spice_carrier:
        spice_bonus = - cfg.get("spice_bonus", 0.3)

    return overweight_cost + day_cost + volume_cost + spice_bonus

def assign(participants, items, cfg):
    random.seed(cfg.get("seed", 42))

    spice_carrier = choose_spice_carrier(participants, cfg)
    participants, total_w = compute_targets(participants, items)

    # состояние
    carried_weight = {name: 0.0 for name in participants["name"]}
    days = defaultdict(Counter)
    assignments = []

    # Сортируем: сначала по дню (ранний вперёд), внутри — по весу (тяжёлые вперёд)
    items_sorted = items.sort_values(by=["day", "weight_kg"], ascending=[True, False]).reset_index(drop=True)

    for _, it in items_sorted.iterrows():
        best_name = None
        best_score = float("inf")
        for _, person in participants.iterrows():
            s = score_candidate(person, it, {"carried_weight": carried_weight, "days": days},
                                cfg, spice_carrier)
            if s < best_score:
                best_score = s
                best_name = person["name"]

        # фиксируем выбор
        carried_weight[best_name] += it["weight_kg"]
        days[best_name][it["day"]] += 1
        assignments.append({
            "person": best_name,
            "item": it["item"],
            "weight_kg": it["weight_kg"],
            "day": it["day"],
            "volume": it["volume"],
            "category": it["category"],
        })

    assign_df = pd.DataFrame(assignments)
    return assign_df, participants[["name","strength","target_weight"]], spice_carrier

def save_outputs(assign_df, participants, spice_carrier):
    OUT.mkdir(parents=True, exist_ok=True)
    # длинный формат
    assign_df.to_csv(OUT / "assignments_long.csv", index=False)

    # широкий формат: по 2 колонки на человека: Еда, Вес
    # порядок людей — как в participants
    people = list(participants["name"])

    # строим по максимальному числу строк на человека
    rows_by_person = {p: assign_df[assign_df["person"]==p].copy().reset_index(drop=True) for p in people}
    max_rows = max((len(df) for df in rows_by_person.values()), default=0)

    # подготавливаем колонки
    wide_cols = []
    for p in people:
        wide_cols += [(f"{p}_Еда", f"{p}_Вес")]

    wide_data = []
    for i in range(max_rows):
        row = {}
        for p in people:
            dfp = rows_by_person[p]
            if i < len(dfp):
                row[f"{p}_Еда"] = f'{dfp.loc[i,"item"]} ({dfp.loc[i,"day"]})'
                row[f"{p}_Вес"] = round(float(dfp.loc[i,"weight_kg"]), 3)
            else:
                row[f"{p}_Еда"] = ""
                row[f"{p}_Вес"] = ""
        wide_data.append(row)

    wide_df = pd.DataFrame(wide_data, columns=sum(([f"{p}_Еда", f"{p}_Вес"] for p in people), []))
    wide_df.to_csv(OUT / "assignments_wide.csv", index=False)

    # также сохраним сводку по весу
    summary = assign_df.groupby("person")["weight_kg"].sum().reset_index().rename(columns={"weight_kg":"carried_weight"})
    summary = participants.merge(summary, left_on="name", right_on="person", how="left").fillna({"carried_weight":0.0})
    summary["delta_to_target"] = summary["carried_weight"] - summary["target_weight"]
    summary["is_spice_carrier"] = summary["name"].eq(spice_carrier)
    summary = summary[["name","strength","target_weight","carried_weight","delta_to_target","is_spice_carrier"]]
    summary.to_csv(OUT / "summary_by_person.csv", index=False)

def main():
    participants, items, cfg = load_data()
    assign_df, participants_aug, spice_carrier = assign(participants, items, cfg)
    save_outputs(assign_df, participants_aug, spice_carrier)
    print("Готово. Смотрите файлы в папке 'output/'.")

if __name__ == "__main__":
    main()

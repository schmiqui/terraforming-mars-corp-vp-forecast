import csv
import json
from pathlib import Path


TAG_KEYS = [
    "building", "jovian", "power", "science", "city", "event",
    "space", "earth", "plant", "animal", "microbe",
]


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_int_or_zero(value) -> int:
    # accepts int/str; invalid like "1/" -> 0
    try:
        s = str(value).strip()
        return int(s) if s.isdigit() else 0
    except Exception:
        return 0


def empty_parameters():
    return {
        "tags": {k: 0 for k in TAG_KEYS},
        "totalVp": 0,
        "totalPrice": 0,
    }


def build_lookup(items):
    # case-insensitive name -> item
    return {item["name"].strip().lower(): item for item in items}


def apply_item(parameters, item):
    if not item:
        return
    parameters["totalVp"] += to_int_or_zero(item.get("vp"))
    parameters["totalPrice"] += to_int_or_zero(item.get("price"))

    for tag in item.get("tags", []):
        if tag in parameters["tags"]:
            parameters["tags"][tag] += 1


def add_card_attributes_to_games(games, cards, corporations):
    card_by_name = build_lookup(cards)
    corp_by_name = build_lookup(corporations)

    for game in games:
        params = empty_parameters()

        for card_name in game.get("cards", []):
            apply_item(params, card_by_name.get(card_name.strip().lower()))

        apply_item(params, corp_by_name.get(game.get("corporation", "").strip().lower()))

        game["parameters"] = params


def export_games_csv(games, out_path="games.csv"):
    tag_keys = TAG_KEYS  # fixed known tags; stable column order
    fieldnames = ["game_id", "points"] + tag_keys + ["totalVp", "totalPrice"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for game in games:
            params = game.get("parameters", {})
            tags = params.get("tags", {})

            writer.writerow({
                "game_id": game.get("game_id", ""),
                "points": game.get("points", ""),
                **{k: tags.get(k, 0) for k in tag_keys},
                "totalVp": params.get("totalVp", 0),
                "totalPrice": params.get("totalPrice", 0),
            })


if __name__ == "__main__":
    base = Path("/Users/jakubsmihula/PycharmProjects/ML/Project/terraforming-mars-corp-vp-forecast")

    cards = read_json(base / "cards/cards.json")
    corporations = read_json(base / "cards/corporations.json")
    games = read_json(base / "datasets/parsed_dataset.json")

    add_card_attributes_to_games(games, cards, corporations)
    export_games_csv(games, "games.csv")
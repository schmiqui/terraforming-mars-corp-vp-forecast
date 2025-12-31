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

        params["corporation"] = game.get("corporation", "").strip().lower()

        game["parameters"] = params


def export_games_csv(games, out_path="games.csv"):
    tag_keys = TAG_KEYS
    fieldnames = ["game_id", "points"] + tag_keys + ["totalVp", "totalPrice", "corporation"]

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
                "corporation": params.get("corporation", ""),
            })

# todo - what if two cards has same name?
def export_games_card_names_csv(games, cards, out_path="games_names.csv"):
    card_names = []
    for item in cards:
        name = item["name"].strip().lower()
        card_names.append(name)

    fieldnames = ["game_id", "points"] + card_names + ["corporation"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for game in games:
            row = {}

            row["game_id"] = game.get("game_id", "")
            row["points"] = game.get("points", 0)

            for name in card_names:
                row[name] = 0

            game_cards = game.get("cards", [])

            normalized_game_cards = []
            for c in game_cards:
                normalized_game_cards.append(str(c).strip().lower())
            for c in normalized_game_cards:
                if c in row:
                    row[c] = 1

            row["corporation"] = game.get("corporation", "").strip().lower()
            writer.writerow(row)

if __name__ == "__main__":
    base = Path("/Users/jakubsmihula/PycharmProjects/ML/Project/terraforming-mars-corp-vp-forecast")

    cards = read_json(base / "cards/cards.json")
    corporations = read_json(base / "cards/corporations.json")
    games = read_json(base / "datasets/parsed_dataset.json")

    export_games_card_names_csv(games, cards, base / "datasets/games_names.csv")

    add_card_attributes_to_games(games, cards, corporations)
    export_games_csv(games, "games.csv")
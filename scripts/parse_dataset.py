import json
import os


def read_dataset(filepath):
    """Load JSON dataset from disk."""
    f = open(filepath, "r")
    try:
        return json.load(f)
    finally:
        f.close()


def parse_games(dataset):
    """Parse both VP totals and card/corporation info, then merge by game_id."""
    vp = parse_vp(dataset.get("gamedict", {}))
    cards = parse_cards(dataset.get("logdict", {}))

    cards_by_id = {}
    for g in cards:
        cards_by_id[g["game_id"]] = g

    out = []
    for v in vp:
        out.append(merge_two_by_game_id(v, cards_by_id.get(v["game_id"], {})))
    return out


def parse_vp(gamedict):
    """Extract VP total per game."""
    out = []
    for game_id, game_data in gamedict.items():
        vp_total = int(game_data["VP total"]["0"])
        out.append({"game_id": game_id, "points": vp_total})
    return out


def parse_cards(logdict):
    """Extract corporation and bought cards per game from log text."""
    out = []
    for game_id, log_text in logdict.items():
        lines = log_text.splitlines()
        out.append(
            {
                "game_id": game_id,
                "corporation": get_chosen_corporation(lines),
                "cards": get_chosen_play_cards(lines),
            }
        )
    return out


def _first_block_after_prefix(lines, prefix):
    """
    Return the first contiguous block of lines starting with `prefix`,
    with the prefix removed and whitespace trimmed.
    """
    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith(prefix):
            start = i
            break

    if start is None:
        return []

    out = []
    i = start
    while i < len(lines) and lines[i].lstrip().startswith(prefix):
        text = lines[i].lstrip()
        out.append(text[len(prefix) :].strip())
        i += 1
    return out


def get_chosen_play_cards(lines):
    return _first_block_after_prefix(lines, "You buy ")


def get_chosen_corporation(lines):
    corps = _first_block_after_prefix(lines, "You choose corporation ")
    if corps:
        return corps[0]
    return None


def merge_two_by_game_id(a, b):
    """Merge two dicts with matching game_id; b fills empty values in a."""
    if b and a.get("game_id") != b.get("game_id"):
        raise ValueError("Different game_id, cannot merge")

    out = dict(a)

    for k, v in b.items():
        if k == "game_id":
            continue

        if k not in out or out[k] in (None, "", [], {}):
            out[k] = v
        elif isinstance(out[k], list) and isinstance(v, list):
            out[k] = out[k] + v
        elif isinstance(out[k], dict) and isinstance(v, dict):
            merged = dict(out[k])
            merged.update(v)
            out[k] = merged
    return out


def save_dataset(data, out_path):
    folder = os.path.dirname(out_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    f = open(out_path, "w")
    try:
        json.dump(data, f, ensure_ascii=False, indent=2)
    finally:
        f.close()


def main():
    in_path = "/Users/jakubsmihula/PycharmProjects/ML/Project/terraforming-mars-corp-vp-forecast/datasets/games.json"
    out_path = "/Users/jakubsmihula/PycharmProjects/ML/Project/terraforming-mars-corp-vp-forecast/datasets/parsed_dataset.json"

    dataset = read_dataset(in_path)
    games = parse_games(dataset)
    save_dataset(games, out_path)
    print(games)


if __name__ == "__main__":
    main()
from typing import List, Tuple, Dict

import pandas as pd
from rapidfuzz import fuzz, process


def build_search_index(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """Create a simple index of (display, track_id, key) for fuzzy search.

    `display` is what we match against, combining track name and artist.
    `key` is the same string used as the lookup key in fuzzy matching.
    """
    records: List[Tuple[str, str, str]] = []
    for _, row in df.iterrows():
        name = str(row.get("track_name", "")).strip()
        artist = str(row.get("track_artist", "")).strip()
        display = f"{name} - {artist}" if artist else name
        records.append((display, str(row.get("track_id", "")), display))
    return records


def fuzzy_find_track_ids(
    user_query: str,
    index: List[Tuple[str, str, str]],
    limit: int = 5,
    score_cutoff: int = 70,
) -> List[Dict[str, str]]:
    """Fuzzy match user text to candidate tracks.

    Returns a list of {display, track_id, score} sorted by score desc.
    """
    # candidates is list of display strings
    candidates = [rec[0] for rec in index]
    matches = process.extract(
        user_query,
        candidates,
        scorer=fuzz.WRatio,
        limit=limit,
        score_cutoff=score_cutoff,
    )

    display_to_track = {display: track_id for display, track_id, _ in index}
    results: List[Dict[str, str]] = []
    for display, score, _idx in matches:
        results.append({
            "display": display,
            "track_id": display_to_track.get(display, ""),
            "score": score,
        })
    return results



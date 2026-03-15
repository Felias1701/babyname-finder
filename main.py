import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

DATA_FILE = Path("data/names.json")


def load_data() -> dict:
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        default = {"names": [], "ratings": {}, "users": [], "new_names": [], "runoffs": []}
        save_data(default)
        return default
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        default = {"names": [], "ratings": {}, "users": [], "new_names": [], "runoffs": []}
        save_data(default)
        return default


def save_data(data: dict) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class NamesPayload(BaseModel):
    names: list[str]


class RatingsPayload(BaseModel):
    ratings: dict[str, dict]  # values: {rating: int | null} or legacy {star, heart}


class UsersPayload(BaseModel):
    users: list[str]
    gender: str | None = None  # "boy" or "girl"


class RunoffPayload(BaseModel):
    names: list[str]


class VotePayload(BaseModel):
    user: str
    winner: str
    loser: str


def normalize_rating(r) -> int | None:
    """Convert rating to int (1-5) or None (unrated).
    Handles both new {rating: int} format and legacy {star, heart} format."""
    if r is None:
        return None
    if isinstance(r, (int, float)):
        v = int(r)
        return v if v in (1, 2, 3, 4, 5) else None
    if isinstance(r, dict):
        if "rating" in r:
            v = r["rating"]
            if v is None:
                return None
            return int(v) if int(v) in (1, 2, 3, 4, 5) else None
        # Legacy {star, heart} format
        if r.get("heart"):
            return 5
        if r.get("star"):
            return 3
        return None
    return None


def calculate_elo(winner_rating: float, loser_rating: float, k: int = 32) -> tuple[float, float]:
    expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
    expected_loser = 1 - expected_winner
    new_winner = winner_rating + k * (1 - expected_winner)
    new_loser = loser_rating + k * (0 - expected_loser)
    return new_winner, new_loser


def get_rounds_required(n: int) -> int:
    pairs = n * (n - 1) // 2
    return max(5, min(30, pairs))


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/users")
def get_users():
    data = load_data()
    return {"users": data.get("users", []), "gender": data.get("gender")}


@app.post("/api/users")
def post_users(payload: UsersPayload):
    users = [u.strip() for u in payload.users if u.strip()]
    if not (1 <= len(users) <= 6):
        raise HTTPException(status_code=400, detail="1 bis 6 Benutzer erlaubt")
    if payload.gender not in ("boy", "girl", None):
        raise HTTPException(status_code=400, detail="Ungültiges Geschlecht")
    data = load_data()
    data["users"] = users
    data["gender"] = payload.gender
    if "ratings" not in data:
        data["ratings"] = {}
    for u in users:
        if u not in data["ratings"]:
            data["ratings"][u] = {}
    save_data(data)
    return {"ok": True, "users": users, "gender": payload.gender}


@app.get("/api/names")
def get_names():
    data = load_data()
    return {"names": data.get("names", []), "new_names": data.get("new_names", [])}


@app.post("/api/names")
def post_names(payload: NamesPayload):
    data = load_data()
    old_names = set(data.get("names", []))
    new_names_list = [n.strip() for n in payload.names if n.strip()]
    added = [n for n in new_names_list if n not in old_names]
    data["names"] = new_names_list
    current_new = set(data.get("new_names", []))
    current_new.update(added)
    data["new_names"] = list(current_new)
    save_data(data)
    return {"ok": True, "names": data["names"], "new_names": data["new_names"]}


@app.get("/api/ratings/{user}")
def get_ratings(user: str):
    data = load_data()
    if user not in data.get("users", []):
        raise HTTPException(status_code=400, detail="Ungültiger Benutzer")
    return data.get("ratings", {}).get(user, {})


@app.post("/api/ratings/{user}")
def post_ratings(user: str, payload: RatingsPayload):
    data = load_data()
    if user not in data.get("users", []):
        raise HTTPException(status_code=400, detail="Ungültiger Benutzer")
    if "ratings" not in data:
        data["ratings"] = {}
    data["ratings"][user] = payload.ratings
    # Remove from new_names if all users have now rated this name
    users = data.get("users", [])
    new_names = set(data.get("new_names", []))
    to_remove = set()
    for name in list(new_names):
        if all(
            normalize_rating(data["ratings"].get(u, {}).get(name)) is not None
            for u in users
        ):
            to_remove.add(name)
    data["new_names"] = list(new_names - to_remove)
    save_data(data)
    return {"ok": True}


@app.get("/api/results")
def get_results():
    data = load_data()
    names: list[str] = data.get("names", [])
    ratings: dict = data.get("ratings", {})
    users: list[str] = data.get("users", [])

    results = []
    for name in names:
        item: dict = {"name": name}
        for u in users:
            item[u] = normalize_rating(ratings.get(u, {}).get(name))
        results.append(item)

    def sort_key(item):
        vals = [item.get(u) for u in users]
        rated_vals = [v for v in vals if v is not None]
        if not rated_vals:
            return (4, 0)
        combined = sum(rated_vals)
        max_combined = 5 * len(users) if users else 5
        if len(rated_vals) == len(users):
            if combined >= max_combined:
                return (0, -combined)
            if combined >= max_combined * 0.7:
                return (1, -combined)
            if combined >= max_combined * 0.5:
                return (2, -combined)
            return (3, -combined)
        return (3, -combined)

    results.sort(key=sort_key)
    return results


@app.post("/api/suggestions")
def get_suggestions():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API-Schlüssel nicht konfiguriert")

    data = load_data()
    ratings: dict = data.get("ratings", {})
    existing_names = {n.lower() for n in data.get("names", [])}

    favorites: set[str] = set()
    for user_ratings in ratings.values():
        for name, r in user_ratings.items():
            v = normalize_rating(r)
            if v is not None and v >= 3:
                favorites.add(name)

    favorites_str = ", ".join(sorted(favorites)) if favorites else "noch keine Favoriten"

    client = anthropic.Anthropic(api_key=api_key)
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=(
                f"You are a baby name advisor. The parents are expecting "
                f"{'a boy' if data.get('gender') == 'boy' else 'a girl' if data.get('gender') == 'girl' else 'a baby'}. "
                "Based on their favorite names, suggest 10 similar names they might love. "
                "Return ONLY a JSON array of objects with 'name' and 'reason' fields "
                "(one sentence each, in German). No markdown, no explanation."
            ),
            messages=[
                {"role": "user", "content": f"Unsere Favoriten: {favorites_str}"}
            ],
        )
        suggestions = json.loads(message.content[0].text)
        # Filter out names already in the list
        filtered = [s for s in suggestions if (s.get("name") or "").lower() not in existing_names]
        return {"suggestions": filtered}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Ungültige KI-Antwort")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"KI-Fehler: {str(e)}")


# ── Runoff routes ──────────────────────────────────────────────────────────────

@app.get("/api/runoffs")
def get_runoffs():
    data = load_data()
    runoffs = data.get("runoffs", [])
    users = data.get("users", [])
    result = []
    changed = False
    for r in runoffs:
        votes = r.get("votes", {})
        rounds_req = r.get("rounds_required", 10)
        user_progress = {u: len(votes.get(u, [])) for u in users}
        all_done = bool(users) and all(user_progress.get(u, 0) >= rounds_req for u in users)
        new_status = "completed" if all_done else "active"
        if r.get("status") != new_status:
            r["status"] = new_status
            changed = True
        result.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "names": r["names"],
            "name_count": len(r["names"]),
            "rounds_required": rounds_req,
            "user_progress": user_progress,
            "status": new_status,
        })
    if changed:
        save_data(data)
    return result


@app.post("/api/runoffs")
def create_runoff(payload: RunoffPayload):
    names = [n.strip() for n in payload.names if n.strip()]
    if len(names) < 2:
        raise HTTPException(status_code=400, detail="Mindestens 2 Namen erforderlich")
    data = load_data()
    if "runoffs" not in data:
        data["runoffs"] = []
    runoff_id = str(uuid.uuid4())[:8]
    rounds_req = get_rounds_required(len(names))
    runoff = {
        "id": runoff_id,
        "created_at": datetime.utcnow().isoformat(),
        "names": names,
        "rounds_required": rounds_req,
        "votes": {u: [] for u in data.get("users", [])},
        "elo": {n: 1000.0 for n in names},
        "status": "active",
    }
    data["runoffs"].append(runoff)
    save_data(data)
    return runoff


@app.get("/api/runoffs/{runoff_id}")
def get_runoff(runoff_id: str):
    data = load_data()
    for r in data.get("runoffs", []):
        if r["id"] == runoff_id:
            return r
    raise HTTPException(status_code=404, detail="Stichwahl nicht gefunden")


@app.post("/api/runoffs/{runoff_id}/vote")
def post_vote(runoff_id: str, payload: VotePayload):
    data = load_data()
    runoff = next((r for r in data.get("runoffs", []) if r["id"] == runoff_id), None)
    if not runoff:
        raise HTTPException(status_code=404, detail="Stichwahl nicht gefunden")
    if payload.user not in data.get("users", []):
        raise HTTPException(status_code=400, detail="Ungültiger Benutzer")
    if payload.winner not in runoff["names"] or payload.loser not in runoff["names"]:
        raise HTTPException(status_code=400, detail="Ungültige Namen")

    if payload.user not in runoff["votes"]:
        runoff["votes"][payload.user] = []
    runoff["votes"][payload.user].append({
        "winner": payload.winner,
        "loser": payload.loser,
        "ts": datetime.utcnow().isoformat(),
    })

    elo = runoff.get("elo", {n: 1000.0 for n in runoff["names"]})
    w_r = elo.get(payload.winner, 1000.0)
    l_r = elo.get(payload.loser, 1000.0)
    new_w, new_l = calculate_elo(w_r, l_r)
    elo[payload.winner] = new_w
    elo[payload.loser] = new_l
    runoff["elo"] = elo

    users = data.get("users", [])
    rounds_req = runoff.get("rounds_required", 10)
    all_done = bool(users) and all(len(runoff["votes"].get(u, [])) >= rounds_req for u in users)
    if all_done:
        runoff["status"] = "completed"

    save_data(data)
    return {
        "ok": True,
        "votes_count": len(runoff["votes"].get(payload.user, [])),
        "rounds_required": rounds_req,
        "elo": elo,
        "status": runoff.get("status", "active"),
    }


# ── Static files (must be last) ───────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")

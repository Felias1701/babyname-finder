import json
import os
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
        default = {"names": [], "ratings": {}, "users": []}
        save_data(default)
        return default
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        default = {"names": [], "ratings": {}, "users": []}
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
    return {"names": data.get("names", [])}


@app.post("/api/names")
def post_names(payload: NamesPayload):
    data = load_data()
    data["names"] = [n.strip() for n in payload.names if n.strip()]
    save_data(data)
    return {"ok": True, "names": data["names"]}


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
        return {"suggestions": suggestions}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Ungültige KI-Antwort")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"KI-Fehler: {str(e)}")


# ── Static files (must be last) ───────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")

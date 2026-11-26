"""
main = linker using fastAPI to map JSON to frontend

HTTPS REQUEST
 GET = requests a representation of specified resources / only retrieve data 
 POST = submits an entity to resources, changing server side 
1. User plays some songs
   → POST /profile 
   body: { history: [...track_ids] }
   ← { session_id: "abc-123", profile: { personality, explorer_score, ... } }

2. User searches for a song to base recommendations on
   → GET /songs?q=radiohead&limit=20
   ← [{ track_id, track_name, ... }]

3. User picks a song → get recommendations
   → POST /recommend 
   body: { track_id, session_id: "abc-123", top_n: 10 }
   ← { query: {...}, profile_used: true, recommendations: [...] }

4. Frontend highlights the cluster on the scatter plot
   → GET /clusters?max_points=3000&cluster_id=<int>
   ← [{ x, y, cluster, track_name, ... }]

Errors:
   404 unknown track_id / session_id
   422 invalid body ( weights, history track_ids)
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import recc as rec
"""
Loaded ONCE at start, shared by every request
df = Dataframe of cleaned tracks
X_sclaed = (n_tracks, 9) audio featrues
scalar = fitted StandardScalar
km = fitted KM model
pca_coords = (n_tracks, 2) x/y poss for scatter plots
profiles = session_id -> profile dict
"""
state = SimpleNamespace(profiles={})
#built in python class, you can attatch attributes to, class to attatch all data to


#asynccontextmanager allows this to set up and after the yield, complete the function
#In our case, this is just for set up to hold the data for when a request comes in
@asynccontextmanager
async def lifespan(app:FastAPI):
    #preprocess data and load df
    state.df = rec.load_data()
    state.X_scaled, state.scalar = rec.preprocess(state.df)

    try:
        #try finding the model and scalar
        state.km, state.scalar = rec.load_artifacts(state.df)
    except (FileNotFoundError, rec.ModelMismatchError) as e:
        print(f"Retraining model as {e}")
        #retrain KM and save another joblib.model
        state.km  = rec.fit_model(state.X_scaled)
        rec.save_artifacts(state.km, state.scalar, state.df)
    state.pca_coords = rec.compute_pca(state.X_sclaed)
    yield
    #after the server ends, it runs yield

app = FastAPI(title="Music Reccomender API", lifespan =lifespan) #lifespan, tells FastAPI what to run at start and end of server

"""
CORS ( Cross Origin Resource Sharing) = way that the server allows it self to communicate to the browser
Enforced by the browser, not the website
Only enforces which websites can reach the server
"""
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], # lets any website call the API, list full browser name once in prod
    allow_methods = ["GET", "POST"], 
    allow_headers = ["*"], #allows any request headers
)

#from Pydantic: BaseModel lays out what data is expected
#turns raw JSON text into objects that we can use  
class ProfileRequest(BaseModel):
    #Field = attatches a default value
    history: list[str] = Field(...,min_length=1) #track_ids, oldest first
    recency_decay: float = Field(0.92, ge=0.5, le=1.0) # 1.0 plays count equaly

class ReccomendRequest(BaseModel):
    track_id: str # saying that every track_id needs to be a string
    top_n:int = Field(10, ge=1, le=50) #ge : <= and le : >=
    session_id:str | None = None # OPTIONAL ( | ) : from POST /Profile, None = No profile
    theory_weight: float = Field(0.25, ge= 0, le=1)
    profile_weight: float = Field(0.20, ge = 0, le = 1)

#Routes
@app.get("/songs")
def search_songs(q: str = Query(...,min_length = 1), limit: int = Query(20, ge=1, le=100)):
    """
    Searches tracks by name or artists(CASE SENSITIVE)

    returns a list 
    Example:
      {
          "track_id": "abc123", "track_name": "Song", "artists": "Artist",
          "danceability": 0.73, "energy": 0.81, "valence": 0.55, "tempo": 120.0,
          "acousticness": 0.02, "instrumentalness": 0.0, "liveness": 0.11,
          "speechiness": 0.04, "loudness": -5.2,
          "key": 9, "mode": 0, "time_signature": 4,
          "key_name": "A", "mood": "minor"
        }

    """
    q.lower()
    mask = (
        state.df["track_name".str.lower().str.contains(q, regex= False, na= False)
                 | state.df["artists"].str.lower().str.conatins(q, regex=False, na=False)]
    )
    cols = ["track_id", "track_name", "artists"] + rec.CLUSTER_FEATURES + rec.THEORY_FEATURES
    hits = state.df[mask].head(limit)[cols].copy()
    hits["key_name"] = hits["key"].map(rec.key_mapping)
    hits["mood"] = hits["mode"].map({1:"major", 0:"minor"}) # maps finction to every item in a list
    return hits.to_dict("records")
@app.post("/profile")
def build_profile(body: ProfileRequest):
    """
    builds listener profile from hsitory 
    Cache it under session_id
    pass session_id --> POST / reccomend
    returns dictionary / JSON text:
      {
          "session_id": "uuid-string",
          "personality": "conventional",   # from the most-played time signature
          "explorer_score": 0.41,          # 0 = comfort listener, 1 = explorer
          "dominant_time_sig": 4,
          "time_sig_entropy": 0.69,        # higher = more varied rhythms
          "genre_entropy": 1.2,            # higher = more varied genres (0 if no genre column)
          "preferred_mode": "minor",       # "major" or "minor"
          "top_keys": ["A", "C", "E"]      # up to 3, most-played first
        }
    """
    profile = rec.listener_profile(
        history = body.histroy,
        df = state.df,
        scalar = state.scalar,
        recency_decay = body.recency_decay,
    )
    if not profile:
        raise HTTPException(422, "None of the history track_ids were found")
    session_id = str(uuid.uuid4()) #random generated ID, uuid4 = version 4, 128 bits, 32 hex digits 
    state.profiles[session_id] = profile

    modes = profile["preferred_modes"]
    return {
        "session_id": session_id,
        "personality": profile["personality"],
        "explorer_score": profile["explorer_score"],
        "dominant_time_sig": profile["dominant_time_sig"],
        "time_sig_entropy": profile["time_sig_entropy"],
        "genre_entropy": profile["genre_entropy"],
        "preferred_mode": "major" if modes[1] >= modes[0] else "minor",
        "top_keys": [rec.key_mapping[k] for k, _ in profile["preferred_keys"].most_common(3)],
    }

app.post("/reccomend")

app.get("/clusters")


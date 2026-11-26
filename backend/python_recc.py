"""
main = linker using fastAPI to map JSON to frontend
1. User plays some songs
   → POST /profile { history: [...track_ids] }
   ← { session_id: "abc-123", profile: { personality, explorer_score, ... } }

2. User searches for a song to base recommendations on
   → GET /songs?q=radiohead
   ← [{ track_id, track_name, ... }]

3. User picks a song → get recommendations
   → POST /recommend { track_id, session_id: "abc-123", top_n: 10 }
   ← { query: {...}, profile_used: true, recommendations: [...] }

4. Frontend highlights the cluster on the scatter plot
   → GET /clusters?cluster_id=7
   ← [{ x, y, cluster, track_name, ... }]

"""

from contextlib import asyncontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastApi, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import python_recc as rec

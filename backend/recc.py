"""
recommender.py = Hybrid Music Reccomendor Engine


"""

import hashlib
import pandas as pd
import joblib 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from scipy.stats import entropy as scipy_entropy


#Algorithm to reccomend music, based off your profile / similar songs in terms of features
#Your local vibe / artist near you = solution to finding smaller artist
# Instrument Seperation = vocal heavy, bass heavy, guitar heavy, etc...
# chord/harmony = legnths / tempo maps for legnth
#genre comparison = do they listen to the same genre all the time or like to expirement?
#Can also incooperate music theory? reccomending songs in similar keys
#Simliar BPM's?



#Grouping songs by audio features = similarity
CLUSTER_FEATURES = [
    "danceability", # range 0-1, 1 dance
    "energy", # range 0-1, fast,loudm noisy the track is, 1 being the most
    "valence", # 0-1, higher to 1 = more positive / closer to 0 = negative, sad, depressed
    "tempo", # 10-250 bpm, float
    "acousticness", # 0 -1, 1 being the track is high in acousticness
    "instrumentalness", #perdicts if vocals are present, 1 being no vocals
    "liveness", #audience present, 1 being its the live edition, reccomend to users who perfer instrumentala
    "speechiness", # 0-1 of how much spoken words, 0.33-0.66 = rap or genres w/ equal beat and lyrics
    "loudness", # how load track is, -60 - 0 db, amplitude research which genres represent which db ranges
]
THEORY_FEATURES = [
    "key", # Key track is in, -1 - 11, -1 = no key was detected
    "mode", #key is major =1 happy / ulifting, or minor = 0, sader / meloncholy
    "time_signature", # 3-7, example = 3 = 3/4 per beat, how many beats there are per bar
    #Can compare time signatures to genre to see if its expiremental? other a=factorsas well
    # 5/4, 7/4 = unconventional for those like "musical caffeine" , 4/4 conventional in pop/rock/EDM, 3/4 = swaying/waltz
]
ALL_FEATURES = CLUSTER_FEATURES + THEORY_FEATURES

key_mapping = {
    -1:"?",
    0:'C',
    1:'C#',
    2:'D',
    3:'D#',
    4:'E',
    5:'F',
    6:'F#',
    7: 'G',
    8: 'G#',
    9:'A',
    10:'A#',
    11:'B'
}
#Relative key pairs: major -> its relative minor, share the same notes
REL_KEYS = {
    0:9, 1:10, 2:11, 3:0, 4:1, 5:2, 
    6:3, 7:4, 8:5, 9:6, 10:7, 11:8,
}
#Circle of fifths: each keys two nearest neighbors
COF_NEIGHBOURS = {
    0: [5, 7],   1: [6, 8],   2: [7, 9],   3: [8, 10],
    4: [9, 11],  5: [10, 0],  6: [11, 1],  7: [0, 2],
    8: [1, 3],   9: [2, 4],  10: [3, 5],  11: [4, 6],
} 
#Time - Signature personality labels 
#3-7 dictionary, key:int, value:str
TIME_SIG_PERSONALITY = {
    3:"waltz", # 3/4 swaying folk/jazz/classical
    4:"conventional", # 4/4 pop, edm, rock normal
    5:"adventurous",  # 5/4 prog, jazz fusion
    6:"flowing", # 6/8 ballads, compound
    7:"complex", # very expiremental, prog rock/metal
}


N_CLUSTERS = 20
MODEL_PATH = "model.joblib" #converts your trained python object into binary file, instead of re-training every time it is called, trianed for large numpy arrays
DATA_PATH = "datasets/example.csv"

def load_data(path:str = DATA_PATH) -> pd.DataFrame:
    """
    Load in the CSV file and prep -> DataFram 
    """

    df = pd.read_csv(path)
    required = ["track_id", "track_name", "artists"] + ALL_FEATURES # make sure that csv data has these labels on their songs

    #Pre-Processing the csv, change anything from data here
    df = df.dropna(subset=required) # Drop empty 
    df = df.drop_duplicates(subset= "track_id") #subset, specifies what to search through
    df = df[df["tempo"]> 0 ] # make sure they have above a 0 tempo

    df["key_detected"] = df["key"] != -1

    df["loudness_norm"] = (df["loudness"] + 60) / 60 # Normalizing loudness, 0 = siilent , 1 = very loud 

    df = df.reset_index(drop=True) # Reset rows, drops unnesicary
    print(f"Loaded {len(df):,} tracks after cleaning")
    return df

#Pre-Process for K-Means
def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Scaling continous cluster features only
    Euclidean distances in cluster spaces caluclated
    """
    X = df[CLUSTER_FEATURES].values # pandas -> NumPy
    scalar = StandardScaler() #  z = (x-mean) / std, it transforms every feature to have mean = 0 and std = 1
    #caluclates z-score, it makes all ranges are average, equally important compared to the mean and std
    """
    example: tempo: [125.0,95.0, 76.0, 110.0] --> [0.75, -0.62, -1.42,0.0] (Hypothetical mean 110, std = 24)
    where positive = above average for that feature / negative = below average compared to mean
    """
    X_scaled = scalar.fit_transform(X) # from scikit-learn, returns same shape but attatches mean=0 / std =1
    return X_scaled, scalar
# K Means for scatter plots
def fit_model(X_scaled:np.ndarray) -> KMeans:
    km = KMeans( 
        n_clusters = N_CLUSTERS,
        init = "k-means++",
        n_init = 10,
        random_state = 42,
        max_iter = 300,
    )
    km.fit(X_scaled)
    print(f"KMeans fitted - at {N_CLUSTERS} clusters. Inertia at {km.inertia_:.1f}")
    #Inertia is metric used to measure compactness of clusters 
    return km

def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(5, 40)) -> None:
    print("\tInertia")
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_scaled)
        print(f"{k}\t {km.inertia_:1.1f}")

#Listener Profiling, build tatse fingerprint from play history = Weighted feature averaging
def listener_profile(
        history: list[str], # ordered list of track_ids ( oldest ->most recent)
        df: pd.DataFrame,
        recency_decay: float = 0.92,) -> dict: #recent plays weights heavily -> larger number
    rows = [] # to track recency, the amount of time a track pops up 
    for track_id in history:
        match = df[df["track_id"] == track_id]
        if not match.empty:
            rows.append(match.iloc[0]) #iloc = int locator, returns the first row in the panda series 
    if not rows:
        return {}
        
    n = len(rows)
    weights = np.array([recency_decay ** (n-1 - i) for i in range(n)])
    weights /= weights.sum()

    #weighted feature averaging grabbing from recency
    feature_matrix = np.array([[r[f] for f in CLUSTER_FEATURES] for r in rows]) # multiplyigng feature weights based off rows and cols, rows = songs, cols = features
    feature_vector = (feature_matrix * weights[:, None]).sum(axis=0)
    instr_avg = float(feature_vector[CLUSTER_FEATURES.index("instrumentalness")])  # weight balanceer to scale for instrumentalness
    feature_vector = scalar.transform(feature_vector.reshape(1,-1))[0]  # reshaping the array to to match ( n, 1) to match (n, num_features)
    # weights[:, None] multiplies each row by weights to create another column -> (n,1) instead of (n,)


    #key preference, music theory
    preferred_keys: Counter = Counter() #sub class of dict, bulit in 342, counting using accumulating values and keys 
    for r, w in zip(rows, weights):
        if r.get("key_detected", True) and int(r["key"]) != -1:
            preferred_keys[int(r["key"])] += w # appending the keys of the songs into an array to decipher
        
    #Mode Preference, major vs minor
    preferred_modes: dict[int,float] = {0:0.0, 1:0.0} # 0 -> minor / 1 -> major, label: weight
    for r,w in zip(rows, weights):
        preferred_modes[int(r["mode"])] += w # same thing, adding a 1 or 0 depending on track's "mood"

    #time signature preference, 3/4, 4/4 = pop, 5/7 = expiremental
    time_sigs = [int(r["time_signature"]) for r in rows]
    ts_counts  = Counter(time_sigs)
    ts_probs = np.array(list(ts_counts.values()), dtype=float) # create a numpy array from the amount of time sigs
    ts_probs /= ts_probs.sum()  # normalize to sum 1
    ts_entropy = float(scipy_entropy(ts_probs)) # entropy returns float / higher entropy (0.9) -> higher the unperdictablity

    #genre entropy -> needs a "track_genre" column 
    genre_entropy = 0.0
    if "track_genre" in df.columns:
        genres = [r.get("track_genre", "unknown") for r in rows ]
        g_counts = Counter(genres)
        g_probs = np.array(list(g_counts.values()), dtype=float) # create a numpy array from the amount of genres
        g_probs /= g_probs.sum()  # normalize to sum 1
        genre_entropy = float(scipy_entropy(g_probs)) # entropy returns float / higher entropy (0.9) -> higher the unperdictablity
    
    # explorer score: rythm variety + genre variety
    #instrumentalness included
    ts_score = min(ts_entropy / 2.0,1.0) 
    g_score = min(genre_entropy / 3.0,1.0)
    explorer_score = round(ts_score * 0.4 + g_score * 0.4 + instr_avg * 0.2, 3) # weighted average of all 3 scores
    # time_sig = 40% of the weight / genres = 40%  / instrumentalness = 20%

    dominant_ts = ts_counts.most_common(1)[0][0] if ts_counts else 4
    
    return {
        "feature_vector":    feature_vector,
        "preferred_keys":    preferred_keys,
        "preferred_modes":   preferred_modes,
        "time_sig_entropy":  round(ts_entropy, 3),
        "genre_entropy":     round(genre_entropy, 3),
        "explorer_score":    explorer_score,
        "dominant_time_sig": dominant_ts,
        "personality":       TIME_SIG_PERSONALITY.get(dominant_ts, "conventional"),
    }


#Musical Theory Matching  = same key, compatability, BPM, Major/ Minor mood = rule-based logic

def music_theory_score(
        query_row: pd.Series,
        canidate_row:pd.Series,
        profile: dict | None = None,
) -> float:
    #returns a 0-1 for how musically compatible a canidate is w a query song / listener profile
    """
    Scoring breakdown (max 1.0):
        Same key                 +0.30
        Relative key             +0.20  (e.g. C major ↔ A minor)
        Circle of 5ths adjacent  +0.10
        Same mode                +0.20  (both major or both minor)
        Same time signature      +0.15
        Near time signature      +0.05  (differ by 1)
        Profile mood alignment   +0.05  (matches user's major/minor lean)
    """
    score = 0.0
    q_key = int(query_row["key"])
    c_key = int(canidate_row["key"])
    q_mode = int(query_row["mode"])
    c_mode = int(canidate_row["mode"])
    q_time_sig = int(query_row["time_signature"])
    c_time_sig = int(canidate_row["time_signature"])

    #Checks for compatibility from the key in the canidate row(pandas series)
    if q_key != -1 and c_key != -1:
        if q_key == c_key:
            score += 0.30
        elif REL_KEYS.get(q_key) == c_key or REL_KEYS.get(c_key) == q_key:
            score += 0.20
        elif c_key in COF_NEIGHBOURS.get(q_key, []):
            score += 0.10
    #Mode Compatibility Check to see if Major or Minor
    if q_mode == c_mode:
        score += 0.20

    #Time Signature Compatability, 3/4ths, 7ths, etc...
    if q_time_sig == c_time_sig: 
        score += 0.15
    elif abs(q_time_sig - c_time_sig) == 1:
        score += 0.05

    if profile and profile.get("preferred_modes"):
        user_leans_minor = (
            profile["preferred_modes"].get(0,0) > 
            profile["preferred_modes"].get(1,0)
        )
        if user_leans_minor and c_mode == 0:
            score += 0.05
        elif not user_leans_minor and c_mode == 1:
            score += 0.05
    return min(score,1.0)

#Artist Discovery Filter 

def artist_discovery_filter(
        canidates: pd.DataFrame, # 1D labeled array 
        profile: dict | None = None,
        popularity_col: str ="popularity",
        explorer_boost_thresh: float =0.45,
) -> pd.Series:
    """
    Adjusts canidate scores based off listerner type/popularity

    exploerer_score > 0.5 => recc more underground artists
    comfort listeners => recc more popular artists

    Returns a multiplier series aligned to canidates.index(1.0)
    """
    if popularity_col not in canidates.columns:
        return pd.Series(1.0, index=canidates.index) #index = label for the data and replacing with baseline 1.0 score

    pop = canidates[popularity_col].clip(0,100) / 100.0 #clip limits values under 100, then turns them into percentages
    exploerer_score = profile.get("explorer_score", 0.5) if profile else 0.5

    if exploerer_score >= explorer_boost_thresh:
        #low popularity -> higher multiplier
        adjustment = 1.0 +(1.0 -pop) * 0.4
    else:
        adjustment = 1.0 + pop * 0.3 
    return adjustment

#THE RECCOMENDATION ENGINE
def recc(
        song_index: int,
        df:pd.DataFrame,
        X_scaled:np.ndarray,
        km: KMeans,
        profile:dict | None = None,
        top_n:int = 10,
        theory_weight:float =0.25,
        profile_weight: float = 0.2,
) -> pd.DataFrame:
    """
    Hybrid Recc Engine
    1. K-Means clustering retrieval
    2. Cosine Similarity -> audio feature similarity
    3. Profile Similarity -> closeness to listener taste
    4. Music Theory Compatability
    5. Artist Discovery Filter

    Weight Distribution(default): 
        cosine_weight = 1- theory_weight - profile_weight = 0.55
        theory_weight = 0.25
        profile_weight = 0.20

    """
    #K-Means CLustering
    song_cluster = km.labels_[song_index]
    song_vector = X_scaled[song_index]
    query_row = df.iloc[song_index]

    #Labeling the indexes of the songs in same cluster
    cluster_mask  = km.labels_ == song_cluster
    cluster_indices = np.where(cluster_mask)[0]
    cluster_vectors = X_scaled[cluster_mask]

    #Cosine Similarity = measuring the angle between two vectors 
    #.linalg is a linear algebra module in numpy / .norm calculates legnth(magnitude) of a vector
    norms = np.linalg.norm(cluster_vectors, axis=1)
    song_norm = np.linalg.norm(song_vector)
    cosine_sim =  cluster_vectors @ song_vector / (norms * song_norm + 1e-10) # @ = matrix multiplication, 1e-10 = small # to avoid 0s

    #Profile Similarity
    #Goal get weights in the range 0-1
    if profile and "feature_vector" in profile:
        pv = profile["feature_vector"] # float array (9,) for 9 features
        pv_norm = pv / (np.linalg.norm(pv) + 1e-10) # normalization (0-1)
        prof_sim = cluster_vectors @ pv_norm # 1D array by mutliplying w cluster_vectors(2D, (n,9)) w nomralization for each song
        ptp = np.ptp(prof_sim) #peak to peak, difference between max and min values
        prof_sim = (prof_sim - prof_sim.min()) / (ptp if ptp > 0 else 1.0) #min-max normalization, min =0 now / max=1
    else: 
        prof_sim = np.zeros_like(cosine_sim) # would create same shape, data type as cosine_sim but filled with zeros

    #Theory Scores
    canidates_df = df.iloc[cluster_indices].copy() # copying cluster songs into new dataframe
    theory_scores = np.array([
        music_theory_score(query_row, canidates_df.iloc[i], profile) # append the music theory score to 2D numpy array, going throuhg canidates
        for i in range(len(canidates_df))
    ])

    #Diversity Multiplier
    diversity_adj = artist_discovery_filter(canidates_df, profile).values # returns multiplier series 

    cosine_weight = 1.0 - theory_weight - profile_weight
    composite = (
        cosine_weight * cosine_sim + 
        theory_weight * theory_scores + 
        profile_weight * prof_sim
    ) * diversity_adj

    #Sort the query songs and drop
    sorted_pos = np.argsort(-composite) # " - " means in decending order
    sorted_pos = np.array([p for p in sorted_pos if cluster_indices[p] != song_index], dtype=int) # remove the song from query
    top_pos = sorted_pos[:top_n]
    top_indices = cluster_indices[top_pos]

    # Build the results 
    results_cols = ["track_id", "track_name", "artists"] + CLUSTER_FEATURES + THEORY_FEATURES
    results_df = df.iloc[top_indices][results_cols].copy()
    results_df["cosine_similarity"] = cosine_sim[top_pos].round(4)
    results_df["theory_score"] = theory_scores[top_pos].round(4)
    results_df["profile_similarity"] =prof_sim[top_pos].round(4)
    results_df["composite_score"] = composite[top_pos].round(4)

    #Key/ Rythm / Mood
    results_df["key_name"] = results_df["key"].map(key_mapping) #.map takes int and return its value
    results_df["mood"] = results_df["mode"].map({1:"major", 0:"minor"}) # assigns labels, major/ minor
    results_df["rhythm_personality"] = results_df["time_signature"].map(
        lambda ts: TIME_SIG_PERSONALITY.get(int(ts), "unknown") # applying time_signature personality labels
    )
    return results_df.reset_index(drop=True)

# PCA for scatter plot
"""
Principal Component Analysis
Takes Data from with many features, finds directions with most spread/variation, re-projects points in those direction

Making 9D song data -> 2D scatter plot
"""
def compute_pca(X_scaled:np.ndarray) -> np.ndarray:
    """
    Returns (n_songs, 2) for recc scatter plot
    """
    pca = PCA(n_components=2, random_state = 42) # 2 components = 2 directions of variance, 
    return pca.fit_transform(X_scaled) #fit = calculates those 2 directions / transform = gives every song coordinates 

#Persists / Load Artifacts
class ModelMismatchError(Exception):
    """rasied when the model loaded at the moment of training is with different settings / data / features
    """
    

def _compute_fingerprint(df:pd.DataFrame) -> dict:
    """fingerprint Dictionary that identifies:
        -Dataset : SHA -256 of all track_ids in curr order
        -Features : SHA-256 of all cluster features
        -Config : N_Clusters value 
    """
    #Dataset fingerprint
    track_id_blob = "|".join(df["track_id"].astype(str).tolist()).encode()
    dataset_hash = hashlib.sha256(track_id_blob).hexdigest()

    #Feature fingerprint: hash the sorted + joined feature column names
    feature_blob = "|".join(CLUSTER_FEATURES).encode()
    feature_hash = hashlib.sha256(feature_blob).hexdigest()

    return {
        "dataset_hash": dataset_hash,
        "feature_hash": feature_hash,
        "n_clusters": N_CLUSTERS,
        "n_tracks": len(df),
    }
def save_artifacts(
        km: KMeans,
        scalar: StandardScaler,
        df: pd.DataFrame,
        path: str = MODEL_PATH,
) -> None:
    #Save trained model, scalar, fingerprint
    fingerprint = _compute_fingerprint(df)
    payload = {
        "model": km,
        "scalar": scalar,
        "fingerprint": fingerprint,
    }
    joblib.dump(payload, path) # the saved model as a binary buffer due to joblib:  a copy of the model
    #.dump() saves model in disk-efficient manner / .load() grabs from memory-> python object that can be used again
    print( f" Artifacts saved in : {path}")

def load_artifacts(
        df:pd.DataFrame,
        path: str = MODEL_PATH,
) -> tuple[KMeans, StandardScaler]:
    """
    Load saved artifacts, raises errors if something not found:
        - FileNotFoundError - model.joblib does not exist yet
        - ModelMisMatchError - fingerprint check failed (stale model)
    """
    payload = joblib.load(path)
    saved_fp = payload.get("fingerprint", {})
    current_fp = _compute_fingerprint(df)

    errors = []
    if saved_fp.get("dataset_hash") != current_fp["dataset_hash"]:
        errors.append(
            f"  Dataset mismatch — saved model was trained on {saved_fp.get('n_tracks', '?'):,} tracks "
            f"(hash: {saved_fp.get('dataset_hash', '?')[:16]}…) "
            f"but current CSV has {current_fp['n_tracks']:,} tracks "
            f"(hash: {current_fp['dataset_hash'][:16]}…)."
        )

    if saved_fp.get("feature_hash") != current_fp["feature_hash"]:
        errors.append(
            f"  Feature mismatch — CLUSTER_FEATURES has changed since the model was saved. "
            f"Saved hash: {saved_fp.get('feature_hash', '?')[:16]}… "
            f"Current hash: {current_fp['feature_hash'][:16]}…"
        )

    if saved_fp.get("n_clusters") != current_fp["n_clusters"]:
        errors.append(
            f"  Cluster count mismatch — saved model used N_CLUSTERS={saved_fp.get('n_clusters', '?')} "
            f"but current config is N_CLUSTERS={current_fp['n_clusters']}."
        )

    if errors:
        raise ModelMismatchError(
            "model.joblib is out of sync with the current data/config. "
            "Re-run `python recommender.py` to retrain.\n"
            + "\n".join(errors)
        )

    print(
        f"Artifacts loaded from {path} ✓\n"
        f"  tracks: {current_fp['n_tracks']:,}  |  "
        f"clusters: {current_fp['n_clusters']}  |  "
        f"hash: {current_fp['dataset_hash'][:16]}…"
    )
    return payload["model"], payload["scalar"]

def build(data_path: str = DATA_PATH):
    df = load_data(data_path)
    X_scaled, scalar = preprocess(df)
    km = fit_model(X_scaled)
    save_artifacts(km, scalar, df) #df now passed so fingerprint is stored
    return df, X_scaled, km, scalar



#Artist Discovery(Local Vibe) = surface lesser-known artists in the same clusters = popularity score filter
""" Work on once main recc is working artist_discovery_filter = surfaces lower-popularity artist
Need additional data, city tag from spotify artist API
"""

# Exploration vs Comfort score = do they stay in one genre or hop around = entropy calculation over histroy = every once in a while throw in a curveball / switch up the data




if __name__ == "__main__":
    df, X_scaled, km, scalar= build()
    recs = recc(0, df, X_scaled, km, top_n = 5)
    print(f"\nReccomendations for : {df.iloc[0]['track_name']}")
    print(recs[["track_name", "artists", "key_name", "mood",
                "rhythm_personality", "composite_score"]].to_string(index=False)) # Pandas Function : doesn't display row index(surrogate)
    mock_history = df["track_id"].iloc[:10].tolist()
    profile = listener_profile(mock_history, df, scalar)
    print(f"\n Listener profile - personality: {profile['personality']}, "
          f"exploerer score : {profile['explorer_score']}")
    recs_p = recc(0, df, X_scaled, km, profile = profile, top_n=5)
    print(f"\n Profile-aware reccomendations for: {df.iloc[0]['track_name']}")
    print(recs_p[["track_name", "artists", "key_name", "mood",
                  "rhythm_personality", "composite_score"]].to_string(index=False))
      


    

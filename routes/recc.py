"""
recommender.py = Hybrid Music Reccomendor Engine


"""

import hashlib
import pandas as pd
import joblib 
import numpy as np
from sklearn.preprocessing import StandardScalar
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
CLUSTER_FEATURES = {
    "daneability", # range 0-1, 1 dance
    "energy", # range 0-1, fast,loudm noisy the track is, 1 being the most
    "valence", # 0-1, higher to 1 = more positive / closer to 0 = negative, sad, depressed
    "tempo", # 10-250 bpm, float
    "acousticness", # 0 -1, 1 being the track is high in acousticness
    "instrumentalness", #perdicts if vocals are present, 1 being no vocals
    "liveness", #audience present, 1 being its the live edition, reccomend to users who perfer instrumentala
    "speechiness", # 0-1 of how much spoken words, 0.33-0.66 = rap or genres w/ equal beat and lyrics
    "loudness", # how load track is, -60 - 0 db, amplitude research which genres represent which db ranges
}
THEORY_FEATURES = {
    "key", # Key track is in, -1 - 11, -1 = no key was detected
    "mode", #key is major =1 happy / ulifting, or minor = 0, sader / meloncholy
    "time_signature", # 3-7, example = 3 = 3/4 per beat, how many beats there are per bar
    #Can compare time signatures to genre to see if its expiremental? other a=factorsas well
    # 5/4, 7/4 = unconventional for those like "musical caffeine" , 4/4 conventional in pop/rock/EDM, 3/4 = swaying/waltz
}
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
    df.dropna(subset=required) # Drop empty 
    df.drop_duplicates(subset= "track_id") #subset, specifies what to search through
    df = df[df["tempo"]> 0 ] # make sure they have above a 0 tempo

    df["key_detected"] = df["key"] != -1

    df["loudness_norm"] = (df["loudness"] + 60) / 60 # Normalizing loudness, 0 = siilent , 1 = very loud 

    df = df.reset_index(drop=True) # Reset rows, drops unnesicary
    print(f"Loaded {len(df):,} tracks after cleaning")
    return df

#Pre-Process for K-Means
def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, StandardScalar]:
    """
    Scaling continous cluster features only
    Euclidean distances in cluuster spaces caluclated
    """
    X = df[CLUSTER_FEATURES].values # pandas -> NumPy
    scalar = StandardScalar() #  z = (x-mean) / std, it transforms every feature to have mean = 0 and std = 1
    #caluclates z-score, it makes all ranges are average, equally important compared to the mean and std
    """
    example: tempo: [125.0,95.0, 76.0, 110.0] --> [0.75, -0.62, -1.42,0.0] (Hypothetical mean 110, std = 24)
    where positive = above average for that feature / negative = below average compared to mean
    """
    X_scaled = scalar.fit_transformation(X) # from scikit-learn, returns same shape but attatches mean=0 / std =1
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
    return km

def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(5, 40)) -> None:
    print("\tInertia")
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_scaled)
        print(f"{k}\t {km.inertia_:1.1f}")

#Listener Profiling, build tatse fingerprint from play history = Weighted feature averaging
def lister_profile(
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
    feature_vector = (feature_matrix * weights[:, None]).sum(axis=0)  # reshaping the array to to match ( n, 1) to match (n, num_features)
    # weights[:, None] multiplies each row by weights to create another colum -> (n,1) instead of (n,)


    #key preference, music theory
    preferred_keys: Counter = Counter() #sub class of dict, bulit in 342, counting using accumulating values and keys 
    for r, w in zip(rows, weights):
        if r.get("key_detected", True) and int(r["key"]) != -1:
            preferred_keys[int(r["key"])] += w # appending the keys of the songs into an array to decipher
        
    #Mode Preference, major vs minor
    preferred_modes: dict[int,float] = {0:0.0, 1:0.0} # 0 -> minor / 1 -> major, label: weight
    for r,w in zip(rows, weights):
        preferred_modes[int(r["modes"])] += w # same thing, adding a 1 or 0 depending on track's "mood"

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
    instr_avg = float(feature_vector[CLUSTER_FEATURES.index("instrumentalness")]) # grab the instrumentalness feature
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

#Artist Discovery(Local Vibe) = surface lesser-known artists in the same clisters = popularity score filter
""" Work on once main recc is working artist_discovery_filter = surfaces lower-popularity artist
Need additional data, city tag from spotify artist API
"""

# Exploration vs Comfort score = do they stay in one genre or hop around = entropy calculation over histroy = every once in a while throw in a curveball / switch up the data




if __name__ == "__main__":
    example_var  = 0#df.iloc[0]["trackname" 
    print(f"\n Sample reccomendations for: {example_var}")

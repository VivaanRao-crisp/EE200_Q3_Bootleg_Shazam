import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import maximum_filter

# all the knobs in one place. this engine is the single source of truth: build_database.py,
# app.py and the notebook all import it, so the database and the queries are always
# fingerprinted with these exact same params. just remember if you change any of these,
# rebuild database.pkl (run build_database.py) or old hashes wont line up with new queries.

NPERSEG = 1024
NOVERLAP = 512
NEIGHBORHOOD = (20, 20)
PEAK_PERCENTILE = 85
TARGET_WINDOW_TIME = 50
TARGET_WINDOW_FREQ = 30
TARGET_SR = 22050  # librosa default, but set it explicitly so its clear that the db and queries are fingerprinted at the same sample rate

# we define functions to load audio, compute spectrogram, find peaks, get hashes, match hashes, and identify songs based on the hashes.
# full pipeline: audio -> spectrogram -> peaks -> paired hashes

def load_audio(path, duration=None):
    # sr=TARGET_SR forces librosa to resample everything to the same sample rate, so the hashes are always fingerprinted identically (whether building the db or querying). mono = 1D signal.
    # duration=None loads the whole file (what build_database.py wants for full tracks). queries pass duration=60 so only ever decode the first 60s — bounds memory and avoids the spectrogram OOM, while accepting clips of any length.
    audio_data, sample_rate = librosa.load(path, sr=TARGET_SR, mono=True, duration=duration)
    return audio_data, sample_rate


def compute_spectrogram(audio_data, sample_rate, nperseg=NPERSEG, noverlap=NOVERLAP):
    # ordinary stft, one little dft per window slice
    frequencies, times, stft = signal.spectrogram(audio_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    # db scale so the quiet stuff and loud stuff are on the same map. 1e-10 dodges log(0), but not too big that it supresses the quiet stuff as observed in notebook.
    stft_db = 10 * np.log10(1e-10 + stft)
    return frequencies, times, stft, stft_db


def find_peaks(stft_db, neighborhood=NEIGHBORHOOD, pct=PEAK_PERCENTILE):
    # a pixel is a peak if its the biggest one in its neighborhood box
    local_max = maximum_filter(stft_db, size=neighborhood) == stft_db
    # and only keep the loud ones so that background noise isn't fingerprinted
    threshold = np.percentile(stft_db, pct)
    peak_mask = local_max & (stft_db > threshold)
    # row = freq index, col = time index
    peak_freq_idx, peak_time_idx = np.where(peak_mask)
    return peak_freq_idx, peak_time_idx # these are the (arrays) coords of the peaks, which will be glued together into pairs and hash in get_hashes()


def get_hashes(audio_data, sample_rate):
    # returns a list of (hash_pair, absolute_time) tuples
    frequencies, times, stft, stft_db = compute_spectrogram(audio_data, sample_rate)
    peak_freq_idx, peak_time_idx = find_peaks(stft_db)

    # glue the coords together and sort by time so that code can scan forwards
    points = list(zip(peak_time_idx, peak_freq_idx))
    points.sort(key=lambda x: x[0]) #sort the list by time index (the first element of the tuple)

    song_hashes = [] #empty list to store the resulting hashes and their corresponding absolute times

    # every point gets a turn as the anchor
    for i in range(len(points)):
        anchor = points[i]

        # look at the points just after it to find a partner
        for j in range(i + 1, len(points)):
            target = points[j]

            time_diff = target[0] - anchor[0]
            freq_diff = abs(target[1] - anchor[1])

            # when time gap too big, reject
            if time_diff > TARGET_WINDOW_TIME:
                break

            # close in frequency, acceptable time gap, make a hash
            if freq_diff <= TARGET_WINDOW_FREQ:
                # hash = (anchor freq, target freq, time gap)
                hash_pair = (anchor[1], target[1], time_diff)
                # turn the anchor's column index back into real seconds
                absolute_time = times[anchor[0]] 
                #anchor[0] is the time index, times[anchor[0]] gives the actual time in seconds
                song_hashes.append((hash_pair, absolute_time))

    return song_hashes


def match(query_hashes, master_db):
    # vote on (song, offset) pairs. a real match dumps tons of votes onto one single offset
    matches = {} #initialize an empty dictionary to store the matches found between the query hashes and the master database
    for q_hash, q_time in query_hashes: # for each hash in the query, see if it exists in the master database
        if q_hash in master_db:
            for db_song_name, db_time in master_db[q_hash]:
                # if its the right song the recording started at a fixed offset, so
                # db_time - q_time should be roughly constant for every matching hash
                offset = round(db_time - q_time, 1)
                key = (db_song_name, offset) #define a 2d kinda key
                # if havent seen this (song, offset) before start it at 0 votes, then add a vote
                if key not in matches:
                    matches[key] = 0
                matches[key] += 1

    if not matches:
        return None

    best_key = max(matches, key=matches.get) #directs max func to use the values obtained by using the get function.
    winning_song, winning_offset = best_key
    winning_score = matches[best_key]

    # pull out the offsets for just the winning song so that histogram can be plotted 
    offset_histogram = {}
    for key in matches:
        song = key[0]
        offset = key[1]
        if song == winning_song:
            offset_histogram[offset] = matches[key] #matches[key] is the number of votes for that (song, offset) pair, so store it in the histogram keyed by offset

    return {
        "song": winning_song,
        "offset": winning_offset,
        "score": winning_score,
        "offset_histogram": offset_histogram,
    }


def identify(path, master_db, duration=None):
    #wrapper function kind of.
    #load audio (capped at `duration` seconds for queries), then get hashes, then match against the master database
    audio_data, sample_rate = load_audio(path, duration=duration)
    query_hashes = get_hashes(audio_data, sample_rate)
    return match(query_hashes, master_db)

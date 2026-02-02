#!/usr/bin/env python3
# dllrc.py

import os
import sys
import time
import json
import argparse
import sqlite3
import threading
import re
from typing import Optional, List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from mutagen import File as MutagenFile
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis
from mutagen.oggopus import OggOpus
from mutagen.mp4 import MP4
from mutagen.id3 import ID3, ID3NoHeaderError, USLT, Encoding
from langdetect import detect_langs

SUPPORTED_EXTENSIONS = {
    ".flac", ".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".wv", ".wav"
}

DURATION_TOLERANCE = 3.0

HTTP_HEADERS = {
    "User-Agent": "dllrc/3.0 (LRC fetcher)"
}
HTTP_TIMEOUT = 10

CONFIG_FILENAME = "dllrc_config.json"
CACHE_FILENAME = "dllrc_cache.sqlite3"
MAX_FETCH_THREADS = 8
RATE_LIMIT_SECONDS = 0.5
ENGLISH_MIN_PROB = 0.80

_rate_lock = threading.Lock()
_last_request_time: Dict[str, float] = {}

def rate_limited_get(url: str, **kwargs) -> requests.Response:
    host = ""
    try:
        host = requests.utils.urlparse(url).netloc
    except Exception:
        host = "default"
    with _rate_lock:
        now = time.time()
        last = _last_request_time.get(host, 0.0)
        delta = now - last
        if delta < RATE_LIMIT_SECONDS:
            time.sleep(RATE_LIMIT_SECONDS - delta)
        _last_request_time[host] = time.time()
    return requests.get(url, **kwargs)

def debug(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def is_music_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS

def first_artist(artist_tag: Optional[str]) -> Optional[str]:
    if not artist_tag:
        return None
    for sep in [";", ",", "/", "&", " feat.", " ft. ", " featuring "]:
        if sep in artist_tag:
            return artist_tag.split(sep)[0].strip()
    return artist_tag.strip()

def clean_atmos_tags(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    suffixes = [
        " (Dolby Atmos Mix)",
        " (Dolby Atmos Edition)"
    ]
    for s in suffixes:
        if text.endswith(s):
            text = text[: -len(s)]
            break
    return text.strip()

def get_tags_and_duration(path: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[float]]:
    try:
        audio = MutagenFile(path, easy=True)
        if audio is None:
            debug(f"Failed to open tags for: {path}")
            return None, None, None, None
        title = None
        album = None
        artist = None
        if "title" in audio and audio["title"]:
            title = str(audio["title"][0])
        if "album" in audio and audio["album"]:
            album = str(audio["album"][0])
        if "artist" in audio and audio["artist"]:
            artist = first_artist(str(audio["artist"][0]))
        duration = None
        if hasattr(audio, "info") and hasattr(audio.info, "length"):
            duration = float(audio.info.length)
        title = clean_atmos_tags(title)
        album = clean_atmos_tags(album)
        return title, album, artist, duration
    except Exception as e:
        debug(f"Error reading tags for {path}: {e}")
        return None, None, None, None

def load_config() -> Dict:
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(script_dir, CONFIG_FILENAME)
        if not os.path.isfile(cfg_path):
            return {}
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            return {}
        return cfg
    except Exception as e:
        debug(f"Error loading config file: {e}")
        return {}

_cache_lock = threading.Lock()
_cache_conn: Optional[sqlite3.Connection] = None

def init_cache(root_folder: str) -> None:
    global _cache_conn
    try:
        db_path = os.path.join(root_folder, CACHE_FILENAME)
        _cache_conn = sqlite3.connect(db_path, check_same_thread=False)
        with _cache_conn:
            _cache_conn.execute("""
                CREATE TABLE IF NOT EXISTS lyrics_cache (
                    artist TEXT,
                    title TEXT,
                    album TEXT,
                    duration REAL,
                    source TEXT,
                    is_synced INTEGER,
                    lyrics TEXT,
                    PRIMARY KEY (artist, title, album, duration, source)
                )
            """)
    except Exception as e:
        debug(f"Error initializing cache: {e}")
        _cache_conn = None
        def cache_get(artist: str, title: str, album: Optional[str], duration: Optional[float]) -> List[Dict[str, Any]]:
    if _cache_conn is None:
        return []
    try:
        with _cache_lock:
            cur = _cache_conn.cursor()
            cur.execute("""
                SELECT source, is_synced, lyrics, duration
                FROM lyrics_cache
                WHERE artist = ? AND title = ? AND (album = ? OR album IS NULL) AND (duration = ? OR duration IS NULL)
            """, (artist, title, album, duration))
            rows = cur.fetchall()
        results = []
        for source, is_synced, lyrics, dur in rows:
            results.append({
                "source": source,
                "is_synced": bool(is_synced),
                "lyrics": lyrics,
                "duration": dur
            })
        return results
    except Exception as e:
        debug(f"Error reading cache: {e}")
        return []

def cache_put(artist: str, title: str, album: Optional[str], duration: Optional[float],
              source: str, is_synced: bool, lyrics: str, lrc_duration: Optional[float]) -> None:
    if _cache_conn is None:
        return
    try:
        with _cache_lock:
            with _cache_conn:
                _cache_conn.execute("""
                    INSERT OR REPLACE INTO lyrics_cache
                    (artist, title, album, duration, source, is_synced, lyrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (artist, title, album, lrc_duration if lrc_duration is not None else duration,
                      source, 1 if is_synced else 0, lyrics))
    except Exception as e:
        debug(f"Error writing cache: {e}")

def strip_lrc_timestamps(text: str) -> str:
    return re.sub(r'\[[0-9:.]+\]', '', text)

def is_english_text(text: str) -> bool:
    try:
        cleaned = text.strip()
        if not cleaned:
            return False
        langs = detect_langs(cleaned)
        if not langs:
            return False
        best = langs[0]
        debug(f"Language detection: {best.lang} ({best.prob:.3f})")
        if best.lang == "en" and best.prob >= ENGLISH_MIN_PROB:
            return True
        return False
    except Exception as e:
        debug(f"Language detection error: {e}")
        return False

def is_english_lyrics(text: str, is_synced: bool) -> bool:
    if is_synced:
        stripped = strip_lrc_timestamps(text)
        return is_english_text(stripped)
    else:
        return is_english_text(text)

def lrclib_search(artist: Optional[str],
                  title: Optional[str],
                  album: Optional[str],
                  duration: Optional[float]) -> Optional[Dict]:
    if not title or not artist:
        return None
    base_url = "https://lrclib.net/api/search"
    params_list: List[Dict] = []
    if album:
        params_list.append({
            "track_name": title,
            "artist_name": artist,
            "album_name": album
        })
    params_list.append({
        "track_name": title,
        "artist_name": artist
    })
    best_result = None
    best_score = None
    for params in params_list:
        try:
            debug(f"LRCLIB search with params: {params}")
            resp = rate_limited_get(base_url, params=params, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
            if resp.status_code != 200:
                debug(f"LRCLIB search HTTP {resp.status_code}")
                continue
            results = resp.json()
            if not isinstance(results, list) or not results:
                debug("LRCLIB search returned no results")
                continue
            for r in results:
                if not r.get("syncedLyrics") and not r.get("plainLyrics"):
                    continue
                track_duration = r.get("duration")
                if track_duration is None:
                    score = 9999.0
                else:
                    try:
                        track_duration = float(track_duration)
                    except Exception:
                        track_duration = None
                    if duration is not None and track_duration is not None:
                        score = abs(duration - track_duration)
                    else:
                        score = 9999.0
                if best_score is None or score < best_score:
                    best_score = score
                    best_result = r
        except Exception as e:
            debug(f"LRCLIB search error: {e}")
            continue
    if best_result and duration is not None and best_result.get("duration") is not None:
        try:
            d = float(best_result["duration"])
            if abs(d - duration) > DURATION_TOLERANCE:
                debug(f"Best LRCLIB result duration mismatch: file={duration:.2f}s, lrc={d:.2f}s")
                return None
        except Exception:
            pass
    return best_result

def extract_lrc_from_lrclib(result: Dict) -> Optional[str]:
    if not result:
        return None
    if result.get("syncedLyrics"):
        return result["syncedLyrics"]
    if result.get("plainLyrics"):
        return result["plainLyrics"]
    return None

def extract_duration_from_lrclib(result: Dict) -> Optional[float]:
    if not result:
        return None
    try:
        if result.get("duration") is not None:
            return float(result["duration"])
    except Exception:
        return None
    return None

def lyricsovh_fetch(artist: Optional[str],
                    title: Optional[str]) -> Optional[str]:
    if not artist or not title:
        return None
    url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
    try:
        debug(f"Lyrics.ovh fetch: {artist} - {title}")
        resp = rate_limited_get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            debug(f"Lyrics.ovh HTTP {resp.status_code}")
            return None
        data = resp.json()
        lyrics = data.get("lyrics")
        if not lyrics:
            return None
        return lyrics.strip()
    except Exception as e:
        debug(f"Lyrics.ovh error: {e}")
        return None
        def chartlyrics_fetch(artist: Optional[str],
                      title: Optional[str]) -> Optional[str]:
    if not artist or not title:
        return None
    url = "http://api.chartlyrics.com/apiv1.asmx/SearchLyricDirect"
    params = {
        "artist": artist,
        "song": title
    }
    try:
        debug(f"ChartLyrics fetch: {artist} - {title}")
        resp = rate_limited_get(url, params=params, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            debug(f"ChartLyrics HTTP {resp.status_code}")
            return None
        text = resp.text
        start_tag = "<Lyric>"
        end_tag = "</Lyric>"
        start = text.find(start_tag)
        end = text.find(end_tag)
        if start == -1 or end == -1 or end <= start:
            return None
        lyrics = text[start + len(start_tag):end]
        lyrics = lyrics.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        lyrics = lyrics.strip()
        if not lyrics:
            return None
        return lyrics
    except Exception as e:
        debug(f"ChartLyrics error: {e}")
        return None

def megalobiz_fetch(artist: Optional[str],
                    title: Optional[str]) -> Optional[str]:
    if not artist or not title:
        return None
    try:
        query = f"{artist} {title}"
        search_url = "https://www.megalobiz.com/search/all"
        params = {"qry": query}
        debug(f"Megalobiz search: {query}")
        resp = rate_limited_get(search_url, params=params, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            debug(f"Megalobiz search HTTP {resp.status_code}")
            return None
        html = resp.text
        m = re.search(r'/lrc/maker/[^"]+', html)
        if not m:
            return None
        lrc_path = m.group(0)
        lrc_url = "https://www.megalobiz.com" + lrc_path
        debug(f"Megalobiz LRC URL: {lrc_url}")
        resp2 = rate_limited_get(lrc_url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if resp2.status_code != 200:
            debug(f"Megalobiz LRC HTTP {resp2.status_code}")
            return None
        html2 = resp2.text
        pre = re.search(r'<textarea[^>]*>(.*?)</textarea>', html2, re.DOTALL | re.IGNORECASE)
        if not pre:
            return None
        lrc = pre.group(1).strip()
        if not lrc:
            return None
        return lrc
    except Exception as e:
        debug(f"Megalobiz error: {e}")
        return None

def lrc123_fetch(artist: Optional[str],
                 title: Optional[str]) -> Optional[str]:
    if not artist or not title:
        return None
    try:
        query = f"{artist} {title}"
        search_url = "https://www.lrc123.com/search"
        params = {"keyword": query}
        debug(f"LRC123 search: {query}")
        resp = rate_limited_get(search_url, params=params, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            debug(f"LRC123 search HTTP {resp.status_code}")
            return None
        html = resp.text
        m = re.search(r'/lrc/\d+\.html', html)
        if not m:
            return None
        lrc_path = m.group(0)
        lrc_url = "https://www.lrc123.com" + lrc_path
        debug(f"LRC123 LRC URL: {lrc_url}")
        resp2 = rate_limited_get(lrc_url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if resp2.status_code != 200:
            debug(f"LRC123 LRC HTTP {resp2.status_code}")
            return None
        html2 = resp2.text
        pre = re.search(r'<textarea[^>]*>(.*?)</textarea>', html2, re.DOTALL | re.IGNORECASE)
        if not pre:
            return None
        lrc = pre.group(1).strip()
        if not lrc:
            return None
        return lrc
    except Exception as e:
        debug(f"LRC123 error: {e}")
        return None

def lyricsify_fetch(artist: Optional[str],
                    title: Optional[str]) -> Optional[str]:
    if not artist or not title:
        return None
    try:
        query = f"{artist} {title}"
        search_url = "https://www.lyricsify.com/search"
        params = {"q": query}
        debug(f"Lyricsify search: {query}")
        resp = rate_limited_get(search_url, params=params, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            debug(f"Lyricsify search HTTP {resp.status_code}")
            return None
        html = resp.text
        m = re.search(r'/lyric/[^"]+', html)
        if not m:
            return None
        lrc_path = m.group(0)
        lrc_url = "https://www.lyricsify.com" + lrc_path
        debug(f"Lyricsify LRC URL: {lrc_url}")
        resp2 = rate_limited_get(lrc_url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if resp2.status_code != 200:
            debug(f"Lyricsify LRC HTTP {resp2.status_code}")
            return None
        html2 = resp2.text
        pre = re.search(r'<textarea[^>]*>(.*?)</textarea>', html2, re.DOTALL | re.IGNORECASE)
        if not pre:
            return None
        lrc = pre.group(1).strip()
        if not lrc:
            return None
        return lrc
    except Exception as e:
        debug(f"Lyricsify error: {e}")
        return None
        def fetch_all_sources(artist: Optional[str],
                      title: Optional[str],
                      album: Optional[str],
                      duration: Optional[float]) -> List[Dict[str, Any]]:
    results = []

    # LRCLIB (synced + plain)
    lrclib_res = lrclib_search(artist, title, album, duration)
    if lrclib_res:
        lrc = extract_lrc_from_lrclib(lrclib_res)
        if lrc:
            lrc_dur = extract_duration_from_lrclib(lrclib_res)
            results.append({
                "source": "lrclib",
                "is_synced": bool(lrclib_res.get("syncedLyrics")),
                "lyrics": lrc,
                "duration": lrc_dur
            })

    # Lyrics.ovh (plain)
    ovh = lyricsovh_fetch(artist, title)
    if ovh:
        results.append({
            "source": "lyricsovh",
            "is_synced": False,
            "lyrics": ovh,
            "duration": None
        })

    # ChartLyrics (plain)
    cl = chartlyrics_fetch(artist, title)
    if cl:
        results.append({
            "source": "chartlyrics",
            "is_synced": False,
            "lyrics": cl,
            "duration": None
        })

    # Megalobiz (synced)
    mega = megalobiz_fetch(artist, title)
    if mega:
        results.append({
            "source": "megalobiz",
            "is_synced": True,
            "lyrics": mega,
            "duration": None
        })

    # LRC123 (synced)
    l123 = lrc123_fetch(artist, title)
    if l123:
        results.append({
            "source": "lrc123",
            "is_synced": True,
            "lyrics": l123,
            "duration": None
        })

    # Lyricsify (synced)
    lyf = lyricsify_fetch(artist, title)
    if lyf:
        results.append({
            "source": "lyricsify",
            "is_synced": True,
            "lyrics": lyf,
            "duration": None
        })

    return results


def choose_best_lyrics(candidates: List[Dict[str, Any]],
                       duration: Optional[float]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    synced = [c for c in candidates if c["is_synced"]]
    plain = [c for c in candidates if not c["is_synced"]]

    if synced:
        if duration is not None:
            synced.sort(key=lambda c: abs((c["duration"] or duration) - duration))
        return synced[0]

    if plain:
        return plain[0]

    return None


def write_lrc_file(path: str, lyrics: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(lyrics)
    except Exception as e:
        debug(f"Error writing LRC file: {e}")


def process_file(path: str, cfg: Dict, root_folder: str) -> None:
    if not is_music_file(path):
        return

    debug(f"Processing: {path}")

    title, album, artist, duration = get_tags_and_duration(path)
    if not title or not artist:
        debug("Missing tags, skipping.")
        return

    cached = cache_get(artist, title, album, duration)
    if cached:
        debug("Using cached lyrics.")
        best = choose_best_lyrics(cached, duration)
        if best:
            out_path = os.path.splitext(path)[0] + ".lrc"
            write_lrc_file(out_path, best["lyrics"])
        return

    fetched = fetch_all_sources(artist, title, album, duration)
    if not fetched:
        debug("No lyrics found.")
        return

    best = choose_best_lyrics(fetched, duration)
    if not best:
        debug("No suitable lyrics found.")
        return

    out_path = os.path.splitext(path)[0] + ".lrc"
    write_lrc_file(out_path, best["lyrics"])

    cache_put(
        artist,
        title,
        album,
        duration,
        best["source"],
        best["is_synced"],
        best["lyrics"],
        best["duration"]
    )


def walk_folder(folder: str, cfg: Dict) -> None:
    init_cache(folder)
    for root, dirs, files in os.walk(folder):
        for name in files:
            full = os.path.join(root, name)
            process_file(full, cfg, folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder to scan for audio files")
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print("Invalid folder.")
        return

    cfg = load_config()
    walk_folder(folder, cfg)


if __name__ == "__main__":
    main()
# Inputs and the web UI

The ADK version read exactly one thing: a YouTube channel, downloaded from
whatever IP the process happened to be running on. That is both the input most
likely to fail and the input fewest people actually have. This version takes
four, resolves them behind one function, and drives all of them from a browser.

---

## Four sources, one resolver

```python
from bookforge.ingest import resolve_source
source = resolve_source(anything, settings)   # → YouTubeSource | LocalVideoSource
                                              #  | TranscriptSource | MixedFolderSource
```

`anything` is a URL, a file, a folder, or a list of those. `run_pipeline` and
`bookforge build` both go through it, so the CLI, the API and the Python
interface accept exactly the same inputs.

| Source | Given | What happens |
|---|---|---|
| **YouTube** | channel / playlist / video URL | list → download → captions or Whisper |
| **Local video** | `.mp4 .mkv .mov .webm .avi .m4v .mpg .mpeg .wmv`, or a folder | ffmpeg probe → captions if a sidecar exists, else Whisper |
| **Transcript** | `.srt .vtt .txt .md`, or a folder | parsed directly — no download, no ffmpeg, no Whisper |
| **Mixed folder** | a folder holding both | videos become chapters; a transcript beside a video of the same stem becomes *that video's* transcript |

```bash
bookforge build ./talks                       # folder — videos, transcripts, or both
bookforge build talk1.mp4 talk2.mp4           # explicit files, in the given order
bookforge build ./transcripts --title "Notes" # text only; fastest path by far
bookforge build "https://youtube.com/@chan"   # as before
```

### Three decisions worth knowing about

**Sidecar pairing.** `talk.mp4` and `talk.srt` in the same folder are one
chapter, not two. The naive glob typesets the talk twice — once from the video
and once from the text — and the duplicate is only obvious after a full run.
Pairing also skips transcription entirely for that video, which is usually the
slowest stage.

**Natural ordering.** Files sort with a numeric-aware key, so `ep10` lands after
`ep2` rather than between `ep1` and `ep2`. Lexicographic sorting silently
reorders any series that passes nine episodes. Set `BOOKFORGE_LOCAL_ORDER` to
`modified` or `created` to order by timestamp instead.

**Stable ids.** A chapter's id is a hash of the file's resolved path, so
re-running over the same folder resumes rather than rebuilding, and a file
reachable by two of the given paths (`build ./talks ./talks/one.mp4`) still
produces one chapter.

### Titles

`01_intro-to-harnesses.mp4` becomes *Intro To Harnesses*: the ordering prefix is
stripped, separators become spaces, and words are title-cased — except when the
filename is already mixed-case, which is taken as deliberate and left alone.
`--title` sets the book title; without it, the folder name is used.

### Transcript formats

SRT, WebVTT, `[00:00:00] text`, and plain prose all parse to the same
`(start_sec, text)` segments.

- VTT scrolling captions repeat the previous line on each cue; duplicates are
  dropped, or every sentence appears twice in the analysis.
- Inline markup (`<i>`, `<c.colorE5E5E5>`) is stripped.
- Plain prose has no timings, and everything downstream is keyed on time —
  chunking, frame captions, cross-references. It gets estimated timings at
  145 wpm. They are wrong in the small and right in the large, which is what
  the windowing actually needs.

---

## YouTube from a server

YouTube refuses media and caption downloads from datacentre IPs:

```
ERROR: [youtube] <id>: Sign in to confirm you're not a bot.
```

Channel *listing* still works, which makes this confusing to debug: intake
succeeds, reports the right number of videos, and then every single download
fails. BookForge recognises the message and prints what to do about it.

```env
BOOKFORGE_YOUTUBE_COOKIES_FILE=/path/to/cookies.txt   # Netscape format
BOOKFORGE_YOUTUBE_COOKIES_FROM_BROWSER=chrome         # desktop only
BOOKFORGE_YOUTUBE_PROXY=http://user:pass@host:port
```

Export `cookies.txt` from a signed-in browser with any of the standard
extensions. Treat the file as a credential: it is a session, and anyone holding
it is signed in as you.

If none of that is available, download the videos on a machine that can and
point BookForge at the folder. That path has no bot check, no rate limit, and
no cookie to rotate.

---

## The web UI

```bash
pip install -e ".[ui]"
bookforge serve                    # http://127.0.0.1:8000
bookforge serve --port 9000 --theme academic
```

Or in Docker, where the UI is the default command:

```bash
docker build -t bookforge .
docker run --rm -p 8000:8000 \
  -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" \
  --env-file .env bookforge
```

The page is one screen: choose a source, preview what was found, start the
build, watch it, open the book.

**Preview before committing.** Pointing at a folder shows the chapters that
would be produced, in order, with their durations, before a single model call
is made. A folder scan that found nothing, or found the files in the wrong
order, is visible in a second rather than an hour.

**Live progress that survives a reconnect.** Events stream over a WebSocket.
The job keeps a bounded replay buffer, so a client that connects mid-build — or
reconnects after a dropped socket, a closed laptop, a phone that switched
networks — receives what it missed instead of an empty log. If WebSockets are
blocked (some corporate proxies), the client falls back to polling the same
events endpoint and nothing about the display changes.

**Verdicts, not just a PDF.** Each finished chapter shows its composite score,
whether it compiled, the defects the verifier found, and the full refinement
history — every attempt, its score, and what the critic said. This is where the
system explains itself; a green PDF with no explanation is exactly what the old
pipeline gave you.

**Cancellation is cooperative.** Cancel takes effect at the next chapter
boundary, so the workspace is never left half-written. Chapters already verified
stay verified, and re-running resumes from there.

### API

| Method | Path | |
|---|---|---|
| `GET` | `/api/health` | readiness, themes, tool availability, effective settings |
| `POST` | `/api/preview` | resolve a source and list what would be built |
| `POST` | `/api/upload` | stage files into a server-side batch folder |
| `POST` | `/api/builds` | start a build |
| `GET` | `/api/builds` · `/api/builds/{id}` | list · one job |
| `GET` |  `/api/builds/{id}/events?after=N` | events after sequence N (the polling fallback) |
| `WS` | `/api/builds/{id}/stream` | live events, with replay from the buffer |
| `POST` | `/api/builds/{id}/cancel` | request cancellation |
| `GET` | `/api/workspaces` · `/{slug}/chapters` | books on disk · per-chapter verdicts |
| `GET` | `/api/workspaces/{slug}/book` · `/{slug}/file?path=` | download |

Interactive docs at `/docs`.

### Before you expose it

**There is no authentication.** `bookforge serve` binds to `127.0.0.1` by
design. The API starts builds, which spend money on model calls, and serves
files from the workspace. Put it behind a proxy that authenticates before
binding it to anything else.

Path handling is defended regardless: every client-supplied path is resolved and
checked to be inside the workspace root before it is opened, and uploaded
filenames are reduced to their basename, so `../../etc/passwd` and
`../../evil.mp4` both go nowhere. Uploads are capped by
`BOOKFORGE_UPLOAD_MAX_MB` (2 GB) per batch — without a ceiling, a browser tab
can fill the disk, and the failure then lands on whatever writes next, usually
a chapter compile in an unrelated build.

One build runs at a time per server. Concurrent builds would race on the
workspace manifest and multiply the model spend without making anything finish
sooner — the pipeline is already prefetching media in parallel.

<div align="center">
<pre>
__  __  _____   ___
\ \/ / |__  /  / _ \
 \  /    / /  | | | |
 /  \   / /_  | |_| |
/_/\_\ /____|  \___/
</pre>

### an exoskeleton for your language model — so it stops forgetting

</div>

---

Every AI model has a memory limit. Talk to it long enough and the early part of the conversation
falls off the end — it forgets the thing you told it an hour ago. **xzo sits between your app and
your model and gives it that memory back.**

You don't retrain anything. You don't switch models. You point your app at xzo instead of at the
model, and long conversations stop hitting the wall.

## What you get

| | |
|---|---|
| 🧠 **It stops forgetting** | When a conversation gets too long, xzo files the older parts away and brings back just the pieces the current turn needs. The model answers as if it still remembered everything. |
| 🔌 **Works with the model you already run** | No retraining, no new model, no changes to your app. If your tool can talk to the OpenAI API, it can talk to xzo. |
| 🤖 **Built for agents too** | Coding assistants and agents that call tools work through it, and long agent sessions keep their facts. |
| 🔒 **Everything stays on your machine** | xzo runs locally against your own model. Nothing is sent anywhere. |
| 🚧 **Conversations can't see each other** | One conversation can never read another's contents, even though they share one store. |

## How it fits together

```
  +------------+       question       +---------+    question + context    +--------------+
  |            |  ----------------->  |         |  --------------------->  |              |
  |  Your app  |                      |   xzo   |                          |  Your model  |
  |            |  <-----------------  |         |  <---------------------  |              |
  +------------+        answer        +----+----+          answer          +--------------+
                                           |
                                           |  files away and looks up
                                           v
                                +----------+---------+
                                |   Memory on disk   |
                                +--------------------+
```

Your app thinks it's talking to the model. The model thinks it's getting a normal, short
conversation. xzo does the remembering in between.

### Why one small program

xzo is a single program: no Python, no virtual environment, no GPU of its own. Small models win on
speed and easy setup, and the layer around them should too. The model stays in whatever engine you
already run; xzo is the thin part around it.

### One rule it keeps: leave the start of the prompt alone

Model servers skip re-reading whatever part of a prompt matches the previous one, and that saving
is most of what keeps each turn fast. So xzo never edits your system prompt at the top of the
conversation; anything it adds goes after it. For the same reason, once a conversation is long, the
line between "summarized" and "kept word for word" moves in steps rather than on every turn, so
most new prompts start exactly like the last one.

## What it does when a conversation gets long

| The situation | What xzo does | What it costs you |
|---|---|---|
| The conversation still fits | Passes it straight through. In the background, it starts summarizing older messages so the work is done before it's needed | Nothing you wait for |
| Too long, and you're after one specific thing | Keeps your instructions and the latest turns word for word, replaces the older part with a short recap, and fetches back the pieces that match | A few seconds |
| Too long, and the question needs the whole thing (*"how many…"*, *"list every…"*) | Skims every stored piece and gathers the result | Slower — it reads everything |

Details worth knowing:

- **Your instructions and your latest message are never changed.** They pass through word for
  word. Only the older middle of the conversation gets filed away.
- **Nothing is summarized twice.** A piece of the conversation that hasn't changed reuses the
  summary already written, even across restarts.
- **Recalled text is marked as notes, not instructions.** Your own earlier requests still count;
  text that came from files or web pages the agent read cannot give the model orders.

### If your app uses tools

Agents call tools, and the results come back as part of the conversation. xzo recognizes this
traffic automatically:

- The **most recent turns stay word for word**, including tool calls and their results. Agents break
  if those are paraphrased.
- Each summarized tool result keeps a label saying which call produced it (for example which file
  was read), so look-alike outputs don't get mixed up.
- A **very large tool result** — a big file, a long command output — is swapped for its first few
  lines plus a note telling the model how to ask for the rest. The full text stays in the store.

## Getting started

**1. Build it.**

```sh
cargo build --release
```

The first run downloads a small model (about 90 MB) that xzo uses to search stored conversation.
That needs an internet connection once; after that it works offline. `cargo build --release
--no-default-features` skips the download, but search gets noticeably worse.

**2. Start your model.** Any OpenAI-compatible server works. With llama.cpp:

```sh
llama-server -m your-model.gguf -c 8192 --reasoning off
```

`-c 8192` is the model's context size. You need it in the next step.

**3. Start xzo, pointed at your model.**

```sh
XZO_CORE_URL=http://127.0.0.1:8080 XZO_NCTX=8192 ./target/release/xzo
```

`XZO_NCTX` must match the model's context size. xzo checks this at startup and tells you if it
doesn't match, and whether it can reach your model at all.

**Using Ollama, MLX or LM Studio instead of llama-server?** Two extra settings:

- `XZO_TOKENIZER=/path/to/tokenizer.json` — the model's tokenizer file, from its Hugging Face page.
  xzo counts tokens with it to know when a conversation no longer fits. llama-server can count for
  xzo; these servers can't, and xzo refuses to start rather than guess.
- Set the server's own context size to match `XZO_NCTX` (Ollama: `num_ctx`). Ollama silently cuts
  the start of a prompt that is too long — where your system prompt is — instead of rejecting it.

**4. Point your app at `http://127.0.0.1:8000`** instead of at the model.

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"xzo","messages":[{"role":"user","content":"hello"}]}'
```

**5. Tell your app the model's window is large.** Many agent tools trim their own history once they
think the model is full. If your app believes the window is 8k, it throws old messages away before
xzo ever sees them. Tell it the model has a large window (for example 128k) and let xzo do the
shrinking.

## Settings

Everything is set through environment variables. These are the ones that matter:

| Setting | Default | What it does |
|---|---|---|
| `XZO_CORE_URL` | `http://127.0.0.1:8080` | where your model is running |
| `XZO_NCTX` | `8192` | **must match your model's context size** |
| `XZO_TOKENIZER` | *(unset)* | path to the model's `tokenizer.json`, so xzo counts tokens itself. Needed for servers without `/tokenize` (Ollama, MLX, LM Studio) |
| `XZO_PORT` | `8000` | the port your app connects to |
| `XZO_HOST` | `127.0.0.1` | which network interface to listen on — **read the security note before changing** |
| `XZO_ALLOW_REMOTE` | `0` | required to listen on anything but this machine. xzo refuses to start without it |
| `XZO_DB` | `xzo_memory.sqlite` | where the conversation store is kept; `none` keeps everything in memory instead |
| `XZO_CHUNK_MAX_ENTRIES` | `50000` | how many pieces of conversation to keep before discarding the oldest |
| `XZO_CHUNK_TTL_DAYS` | `0` | delete stored pieces older than N days when `/prune` runs; `0` keeps them until the cap |
| `XZO_OVERFLOW_TRIGGER` | `0.9` | how full the model's window gets before xzo steps in (0.9 = 90%) |
| `XZO_OVERFLOW_MAX_INFLIGHT` | `4` | how many long requests may be worked on at once; extra ones wait, none are rejected |

`./target/release/xzo --help` lists the rest.

## Where your conversations are stored

xzo writes a file next to itself — `xzo_memory.sqlite` by default. **It contains the text of your
conversations, in plain readable form, and it stays there after you shut xzo down.** If your app
uses tools, that includes **tool results** — the contents of files an agent read, the output of
commands it ran. Treat the file the way you'd treat the conversations themselves.

- To keep nothing on disk, set `XZO_DB=none`. Everything then lives in memory and disappears when
  you exit.
- To clear stored conversations without deleting files by hand, send a request to `/prune`.
- The store won't grow forever: `XZO_CHUNK_MAX_ENTRIES` caps it, at roughly a few hundred megabytes
  by default.

> ### 🔒 Keep it on your own machine
> **xzo has no password or login of any kind.** It listens only on your own computer by default,
> which is the safe setting. Do not open it to your network or the internet unless you put something
> in front of it that handles authentication — otherwise anyone who can reach the port can read your
> stored conversations and wipe your memory file.

## Checking on it

Visit `http://127.0.0.1:8000/stack` for a status readout:

| What you see | What it tells you |
|---|---|
| `chunks.entries` | how many pieces of conversation are stored |
| `chunks.evictions` | how many old pieces were discarded to stay under the cap |
| `chunks.insert_failures` | failed writes — should always be `0`; anything else usually means the disk is full |
| `prewarm_busy` | whether xzo is summarizing in the background right now |
| the `overflow` and `compaction` numbers | how often xzo steps in, and what it did |

Other addresses: `/health` (alive check), `/v1/models`, and `/prune` (clear the store).

## Does it actually work?

Measured with Qwen3.5-9B on a 12 GB graphics card, with an 8,192-token window. "Without xzo" means
the app keeps only the messages that fit, which is what apps do on their own.

**Agent sessions** — a coding agent reads files and runs commands, then is asked about something
from early in the session:

| Session length | Correct with xzo | Correct without xzo |
|---|---|---|
| ~30 turns (20 sessions) | **20 / 20** | 4 / 20 |
| ~85 turns (10 sessions) | **10 / 10** | 2 / 10 |
| ~225 turns (10 sessions) | **10 / 10** | 2 / 10 |

Without xzo, only the questions about the most recent messages get answered. With xzo, facts from
200 turns back are still found.

**Waiting time**, in a 30-turn agent session with a few seconds of agent work between turns:

| | Wait |
|---|---|
| Turns before the conversation outgrows the window | about 1 second, same as without xzo |
| The turn where it outgrows the window | 4–16 seconds |
| Each turn after that | about 5 seconds |

**Document questions** — a long document pasted into the conversation, then questions about it:
**9 / 9** answered correctly with xzo, **0 / 9** without (the model runs out of room). A follow-up
question about the same document took **3.7 seconds instead of 25.2**, because it reuses the work
done for the first.

**Bottom line: long conversations work that otherwise can't.** Two limits on those numbers: one
model was measured, and the samples are small (2–4 per kind of question), enough to see 10/10
against 2/10 but not to rank close results.

## Things to know before you rely on it

- **Tested with one model.** Everything above was measured with Qwen3.5-9B. Other models should
  work but haven't been measured.
- **The first answer on a long history xzo has never seen is slow.** Sent all at once, an 85-turn
  conversation takes several minutes the first time, because every piece has to be stored and the
  newest ones summarized. In a conversation xzo has followed turn by turn, that work is already done.
- **The no-download build is worse.** `--no-default-features` avoids the 90 MB download but uses a
  much cruder way to search stored conversation. It's a fallback, not an equal.
- **"How many…" style questions are detected using English phrases.** Ask that kind of question in
  another language and xzo may treat it as an ordinary lookup.

## One thing to turn off on your model

If your model has a "thinking" or "reasoning" mode, **turn it off**. Several smaller models do
noticeably worse with it on, and some turn it on by default. xzo deliberately doesn't touch this
setting — how you disable it depends on your model. With llama.cpp it's `--reasoning off`.

## Running the tests

```sh
cargo test --no-default-features   # no download needed
cargo test                         # everything
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

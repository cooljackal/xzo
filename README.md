<div align="center">
<pre>
__  __  _____   ___
\ \/ / |__  /  / _ \
 \  /    / /  | | | |
 /  \   / /_  | |_| |
/_/\_\ /____|  \___/
</pre>

**An exoskeleton for your language model — so it stops forgetting.**

</div>

Every model has a memory limit. Talk long enough and the start of the conversation falls off.
**xzo sits between your app and your model and gives it that memory back.**

- 🧠 **Remembers long conversations.** Old messages are filed away; the right pieces come back when needed.
- 🔌 **No changes to your app or model.** Anything that speaks the OpenAI API works.
- 🤖 **Works for agents.** Tool calls and results are handled, not just chat.
- 🔒 **Local only.** Nothing leaves your machine.
- 🚧 **Conversations stay separate.** One can never read another's contents.
- 📦 **One small program.** No Python, no GPU of its own.

```
Your app  ──►  xzo  ──►  Your model
                │
                └──►  memory on disk
```

## Quick start

1. **Build**
   ```sh
   cargo build --release
   ```
   First run downloads a ~90 MB search model, once.

2. **Start your model** (llama.cpp example; `-c` is the context size)
   ```sh
   llama-server -m your-model.gguf -c 8192 --reasoning off
   ```

3. **Start xzo** (`XZO_NCTX` must equal `-c`)
   ```sh
   XZO_CORE_URL=http://127.0.0.1:8080 XZO_NCTX=8192 ./target/release/xzo
   ```

4. **Point your app at `http://127.0.0.1:8000`** instead of the model.

5. **Tell your app the model's window is large** (e.g. 128k). Otherwise many agent tools throw
   old messages away before xzo sees them.

**Ollama, MLX or LM Studio?** Also set:
- `XZO_TOKENIZER=/path/to/tokenizer.json` (from the model's Hugging Face page), so xzo can count tokens.
- The server's context size to match `XZO_NCTX` (Ollama: `num_ctx`). Ollama silently cuts long prompts.

## How it works

| Conversation | What xzo does |
|---|---|
| Still fits | Passes it through. Starts summarizing old messages in the background. |
| Too long, specific question | Keeps your instructions and recent turns word for word, swaps the older part for a short recap, fetches the pieces that match. |
| Too long, "how many / list all" question | Reads every stored piece and gathers the answer. |

- Your instructions and latest message are never changed.
- Nothing is summarized twice, even across restarts.
- Recalled text is marked as notes: your earlier requests still count, but text from files or web pages can't give the model orders.
- Big tool results (files, command output) are cut to a preview plus a "ask for the rest" note.

## Results

Qwen3.5-9B, 12 GB GPU, 8,192-token window. "Without xzo" = the app keeps only what fits.

| Test | With xzo | Without xzo |
|---|---|---|
| Agent sessions, ~30 turns | **20 / 20** | 4 / 20 |
| Agent sessions, ~85 turns | **10 / 10** | 2 / 10 |
| Agent sessions, ~225 turns | **10 / 10** | 2 / 10 |
| Long-document questions | **9 / 9** | 0 / 9 |

Correct answers out of sessions tested. Without xzo, only questions about recent messages get answered.

| Moment in a 30-turn agent session | Wait |
|---|---|
| Before the conversation is too long | ~1 s |
| The turn it gets too long | 4–16 s |
| Every turn after | ~5 s |

Small samples, one model: enough to show the gap, not to rank close results.

## Settings

All settings are environment variables. `xzo --help` lists every one.

| Setting | Default | What it does |
|---|---|---|
| `XZO_CORE_URL` | `http://127.0.0.1:8080` | Where your model runs |
| `XZO_NCTX` | `8192` | **Must match your model's context size** |
| `XZO_TOKENIZER` | — | Model's `tokenizer.json`; needed for Ollama, MLX, LM Studio |
| `XZO_PORT` | `8000` | Port your app connects to |
| `XZO_HOST` | `127.0.0.1` | Interface to listen on (see Security) |
| `XZO_ALLOW_REMOTE` | `0` | Required to listen beyond this machine |
| `XZO_DB` | `xzo_memory.sqlite` | Memory file; `none` = keep nothing on disk |
| `XZO_CHUNK_MAX_ENTRIES` | `50000` | Max stored pieces before the oldest go |
| `XZO_CHUNK_TTL_DAYS` | `0` | Delete pieces older than N days on `/prune` (`0` = never) |
| `XZO_OVERFLOW_TRIGGER` | `0.9` | How full the window gets before xzo steps in |
| `XZO_OVERFLOW_MAX_INFLIGHT` | `4` | Long requests handled at once; extras wait |

## Endpoints

| Address | What it's for |
|---|---|
| `/v1/chat/completions` | Your app talks here |
| `/v1/models` | Model list |
| `/health` | Alive check |
| `/stack` | Status counters (`chunks.insert_failures` should stay `0`) |
| `/prune` | Clear old memory |

## Security

- **No login.** Anyone who can reach the port can read and wipe your memory. xzo listens only on
  your machine by default; don't expose it without something in front that handles authentication.
- **Memory is plain text on disk**, including tool results (files an agent read, command output).
  Use `XZO_DB=none` to keep nothing, or `/prune` to clear it.
- Report problems privately: see [SECURITY.md](SECURITY.md).

## Limits

- Measured with one model (Qwen3.5-9B). Others should work but aren't measured.
- A long history xzo has never seen, sent all at once, takes minutes the first time. Followed turn by turn, it doesn't.
- `--no-default-features` skips the 90 MB download but searches much worse.
- "How many…" questions are detected in English only.
- Turn off your model's "thinking" mode (llama.cpp: `--reasoning off`); small models do worse with it.

## Tests

```sh
cargo test
```

## License

[Apache 2.0](LICENSE)

# Security

## Reporting a vulnerability

Open a draft GitHub security advisory:

<https://github.com/jusso-dev/protect-audio-search/security/advisories/new>

There is no separate security email. Do not file a public issue for a leaked secret or for a bug that can be exploited. Do not include a live password, token, or API key in the report.

## Secrets

- Passwords and API keys come from environment variables. The CLI has no flag for them.
- This package reads `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` only when `--confirm` is building the vision client. It does not print those values. A missing key exits with a message that names `ANTHROPIC_API_KEY`.
- UniFi Protect credentials are read by the child process `npx -y @mgcrea/mcp-unifi-protect@1.1.0`. This package does not parse them. The child gets an allowlist of the environment: `UNIFI_PROTECT_*`, `PATH`, `HOME`, the user and shell names, locale, temp, proxy and certificate variables, and `NODE_*`, `NPM_CONFIG_*`, `XDG_*`, `NVM_*`. `ANTHROPIC_API_KEY` and anything else outside that list are not passed.
- Text this tool prints for its own errors goes through redaction: values next to `password`, `token`, `secret`, `api_key`, `totp`, `authorization`, and `cookie` (plain or JSON-quoted), bearer tokens, credentials inside URLs, and the literal values of `UNIFI_PROTECT_PASSWORD`, `UNIFI_PROTECT_API_KEY`, `UNIFI_PROTECT_TOTP` and `ANTHROPIC_API_KEY`. That is not a guarantee. The MCP SDK can still log a raw tool error on stderr.
- Do not commit `.env`, footage, clips, snapshots, or model weights. `.gitignore` already excludes them. Do not commit a filled-in config for the MCP server either.

## What a scan is allowed to do

The default child is `npx -y @mgcrea/mcp-unifi-protect@1.1.0`. `npx -y` downloads and runs that package. The version is pinned so a cold npx cache does not pull an unreviewed release, but its own dependencies use version ranges. Treat it as third-party code. To run a copy you installed and locked yourself, set `PROTECT_AUDIO_SEARCH_MCP_COMMAND`. The tool calls only `unifi_protect_list_cameras` and `unifi_protect_export_video`, and lists the server's tools first.

Each export names its own destination (`savePath`) inside `--output-dir`. A server that writes anywhere else is refused, and that file is not copied. Partial downloads are removed when an export fails.

Leave `UNIFI_PROTECT_ALLOW_WRITES` unset so that server does not register tools that change the console. Use a local view-only account. Leave TLS verification on. Do not set `UNIFI_PROTECT_VERIFY_TLS=false`. Certificate pinning is the MCP server's job and is described in its README.

## Model files

On first use the TFLite model and the class map are downloaded and checked against pinned SHA-256 hashes. A mismatch raises `ModelHashError` and does not treat the bad file as good.

- Model: `10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de`
- Class map: `cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2`

After the cache is warm, `PROTECT_AUDIO_SEARCH_OFFLINE=1` makes a missing file fail instead of downloading.

## Experimental scan server

`python -m protect_audio_search.search_mcp` is not a supported interface. The decision is in [docs/search-audio-events.md](docs/search-audio-events.md). It uses the same environment and the same two Protect read tools as the CLI. Tool arguments are a camera, a time range, and a timezone, not credentials and not a directory. Footage goes to `exports` under the server's working directory. A failed job stores redacted error text in memory only.

## Data that leaves the machine

A scan without `--confirm` keeps clips and snapshots local. Files it creates are mode `0600` and folders `0700` (the scan sets a `077` umask, which the MCP child inherits). CSV cells that would start a spreadsheet formula are escaped, and text from the vision model has control characters stripped. It still talks to your Protect console (or to whatever host the MCP server is configured to use) and, once, to the model download URLs.

`--confirm` sends snapshot JPEGs to Anthropic. It is off by default. See [docs/vision-confirmation.md](docs/vision-confirmation.md).

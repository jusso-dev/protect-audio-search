# Security

## Reporting a vulnerability

Open a draft GitHub security advisory:

<https://github.com/jusso-dev/protect-audio-search/security/advisories/new>

There is no separate security email. Do not file a public issue for a leaked secret or for a bug that can be exploited. Do not include a live password, token, or API key in the report.

## Secrets

- Passwords and API keys come from environment variables. The CLI has no flag for them.
- This package reads `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` only when `--confirm` is building the vision client. It does not print those values. A missing key exits with a message that names `ANTHROPIC_API_KEY`.
- UniFi Protect credentials are read by the child process `npx -y @mgcrea/mcp-unifi-protect`, which inherits the environment. This package does not parse them.
- Text this tool prints for its own errors goes through redaction for values next to `password`, `token`, `secret`, and `api_key`. That is not a guarantee. The MCP SDK can still log a raw tool error on stderr.
- Do not commit `.env`, footage, clips, snapshots, or model weights. `.gitignore` already excludes them. Do not commit a filled-in config for the MCP server either.

## What a scan is allowed to do

The default child is `npx -y @mgcrea/mcp-unifi-protect`. `npx -y` downloads and runs that package. Treat it as third-party code. The tool calls only `unifi_protect_list_cameras` and `unifi_protect_export_video`.

Leave `UNIFI_PROTECT_ALLOW_WRITES` unset so that server does not register tools that change the console. Use a local view-only account. Leave TLS verification on. Do not set `UNIFI_PROTECT_VERIFY_TLS=false`. Certificate pinning is the MCP server's job and is described in its README.

## Model files

On first use the TFLite model and the class map are downloaded and checked against pinned SHA-256 hashes. A mismatch raises `ModelHashError` and does not treat the bad file as good.

- Model: `10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de`
- Class map: `cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2`

After the cache is warm, `PROTECT_AUDIO_SEARCH_OFFLINE=1` makes a missing file fail instead of downloading.

## Data that leaves the machine

A scan without `--confirm` keeps clips and snapshots local. It still talks to your Protect console (or to whatever host the MCP server is configured to use) and, once, to the model download URLs.

`--confirm` sends snapshot JPEGs to Anthropic. It is off by default. See [docs/vision-confirmation.md](docs/vision-confirmation.md).

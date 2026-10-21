# Vision confirmation

`--confirm` is off unless you set it. With the flag off, this tool does not call an LLM.

When the flag is on, snapshot JPEGs are sent to Anthropic (Claude Haiku by default). Those images leave this machine. Do not use it for frames you are not allowed to share with a third party. Australian surveillance and listening laws vary. This file is not legal advice.

The result is a hint for a person. It is not evidence that a vehicle, smoke, or a number plate was present.

## Configuration

The API key is read from `ANTHROPIC_API_KEY`. An optional model id is read from `ANTHROPIC_MODEL` (default `claude-haiku-5-5`). The tool never prints those values.

`--confirm` needs `--snapshots N`, because the images are the snapshots for each event.

## Caps

Defaults: 3 images per event, 20 calls, and USD 0.50 estimated cost. The estimate uses the published Claude Haiku 5.5 short-prompt rates: USD 0.10 per million input tokens and USD 0.50 per million output tokens. It is not an invoice. A run stops calling once the call cap or the cost cap is reached, and prints a summary of calls, images, estimated cost, and skipped events.

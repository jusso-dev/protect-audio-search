# Test fixtures

## engine_cc0.wav

One 0.975 second window (16 kHz mono 16-bit PCM) cut from `engine_sound_1.mp3`, starting at 7.5 seconds.

- Source: [engine sounds](https://opengameart.org/content/engine-sounds2) (`engine_sound_1.mp3`)
- Author: pauliuw (Paulius Jurgelevičius)
- Licence: CC0 1.0

YAMNet's top class on this window is `Motorcycle`, which is one of the default target classes.

## silence and squeal

The end-to-end test builds these with the stdlib `wave` module. They are not recordings and have no third-party licence.

- Silence: one second of digital zero, 16 kHz mono.
- Squeal: a one second sine sweep from 2 kHz to 6 kHz, 16 kHz mono. It is a stand-in tone, not a tyre recording, and the scan does not treat it as a `Tire squeal` hit.

The mock server muxes a wav into a 3 second MP4 when `MOCK_MEDIA_WAV` is set. The engine scan is expected to report `Motorcycle`. The silence scan is expected to report no events.

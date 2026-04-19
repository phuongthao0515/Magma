Image-based integration tests live here.

How to use:

1. Copy `scenarios.example.json` to `scenarios.json`.
2. Put your screenshots in subfolders here, for example:
   - `one_step/step_0.png`
   - `two_step/step_0.png`
   - `two_step/step_1.png`
   - `two_step_font_size/step_0.png`
   - `two_step_font_size/step_1.png`
3. Fill in the prompt and the expected action for each step in `scenarios.json`.
4. Run:

```bash
FULL_APP_USE_REAL_MODEL_TESTS=1 \
PYENV_VERSION=agent \
pyenv exec python -m pytest tests/integration -m integration -rs
```

The integration tests call the real server-side screenshot pipeline and assert the returned action payload.
If you want to assert the exact normalized PyAutoGUI command, use `expected_pyautogui_call`.

Current fixture set:

- `one_step`: click the `Bold` button
- `two_step`: click the `Bold` button, then type text
- `two_step_font_size`: click the font size dropdown, then click `14`

If you want another distinct two-step test, provide another screenshot pair that leads to a visibly different target or a different UI state.

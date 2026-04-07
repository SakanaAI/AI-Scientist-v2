# Voice dictation

> Use push-to-talk voice dictation to speak your prompts instead of typing them in the Claude Code CLI.

Hold a key and speak to dictate your prompts. Your speech is transcribed live into the prompt input, so you can mix voice and typing in the same message. Enable dictation with `/voice`. The default push-to-talk key is `Space`; [rebind to a modifier combination](#rebind-the-push-to-talk-key) to activate on the first keypress rather than after a brief hold.

<Note>
  Voice dictation requires Claude Code v2.1.69 or later. Check your version with `claude --version`.
</Note>

## Requirements

Voice dictation streams your recorded audio to Anthropic's servers for transcription. Audio is not processed locally. The speech-to-text service is only available when you authenticate with a Claude.ai account, and is not available when Claude Code is configured to use an Anthropic API key directly, Amazon Bedrock, Google Vertex AI, or Microsoft Foundry. See [data usage](/en/data-usage) for how Anthropic handles your data.

Voice dictation also needs local microphone access, so it does not work in remote environments such as [Claude Code on the web](/en/claude-code-on-the-web) or SSH sessions. In WSL, voice dictation requires WSLg for audio access, which is included with WSL2 on Windows 11. On Windows 10 or WSL1, run Claude Code in native Windows instead.

Audio recording uses a built-in native module on macOS, Linux, and Windows. On Linux, if the native module cannot load, Claude Code falls back to `arecord` from ALSA utils or `rec` from SoX. If neither is available, `/voice` prints an install command for your package manager.

## Enable voice dictation

Run `/voice` to toggle voice dictation on. The first time you enable it, Claude Code runs a microphone check. On macOS, this triggers the system microphone permission prompt for your terminal if it has never been granted.

```
/voice
Voice mode enabled. Hold Space to record. Dictation language: en (/config to change).
```

Voice dictation persists across sessions. Run `/voice` again to turn it off, or set it directly in your [user settings file](/en/settings):

```json  theme={null}
{
  "voiceEnabled": true
}
```

While voice dictation is enabled, the input footer shows a `hold Space to speak` hint when the prompt is empty. The hint does not appear if you have a [custom status line](/en/statusline) configured.

## Record a prompt

Hold `Space` to start recording. Claude Code detects a held key by watching for rapid key-repeat events from your terminal, so there is a brief warmup before recording begins. The footer shows `keep holding…` during warmup, then switches to a live waveform once recording is active.

The first couple of key-repeat characters type into the input during warmup and are removed automatically when recording activates. A single `Space` tap still types a space, since hold detection only triggers on rapid repeat.

<Tip>
  To skip the warmup, [rebind to a modifier combination](#rebind-the-push-to-talk-key) like `meta+k`. Modifier combos start recording on the first keypress.
</Tip>

Your speech appears in the prompt as you speak, dimmed until the transcript is finalized. Release `Space` to stop recording and finalize the text. The transcript is inserted at your cursor position and the cursor stays at the end of the inserted text, so you can mix typing and dictation in any order. Hold `Space` again to append another recording, or move the cursor first to insert speech elsewhere in the prompt:

```
> refactor the auth middleware to ▮
  # hold Space, speak "use the new token validation helper"
> refactor the auth middleware to use the new token validation helper▮
```

Transcription is tuned for coding vocabulary. Common development terms like `regex`, `OAuth`, `JSON`, and `localhost` are recognized correctly, and your current project name and git branch name are added as recognition hints automatically.

## Change the dictation language

Voice dictation uses the same [`language` setting](/en/settings) that controls Claude's response language. If that setting is empty, dictation defaults to English.

<Accordion title="Supported dictation languages">
  | Language   | Code |
  | :--------- | :--- |
  | Czech      | `cs` |
  | Danish     | `da` |
  | Dutch      | `nl` |
  | English    | `en` |
  | French     | `fr` |
  | German     | `de` |
  | Greek      | `el` |
  | Hindi      | `hi` |
  | Indonesian | `id` |
  | Italian    | `it` |
  | Japanese   | `ja` |
  | Korean     | `ko` |
  | Norwegian  | `no` |
  | Polish     | `pl` |
  | Portuguese | `pt` |
  | Russian    | `ru` |
  | Spanish    | `es` |
  | Swedish    | `sv` |
  | Turkish    | `tr` |
  | Ukrainian  | `uk` |
</Accordion>

Set the language in `/config` or directly in settings. You can use either the [BCP 47 language code](https://en.wikipedia.org/wiki/IETF_language_tag) or the language name:

```json  theme={null}
{
  "language": "japanese"
}
```

If your `language` setting is not in the supported list, `/voice` warns you on enable and falls back to English for dictation. Claude's text responses are not affected by this fallback.

## Rebind the push-to-talk key

The push-to-talk key is bound to `voice:pushToTalk` in the `Chat` context and defaults to `Space`. Rebind it in [`~/.claude/keybindings.json`](/en/keybindings):

```json  theme={null}
{
  "bindings": [
    {
      "context": "Chat",
      "bindings": {
        "meta+k": "voice:pushToTalk",
        "space": null
      }
    }
  ]
}
```

Setting `"space": null` removes the default binding. Omit it if you want both keys active.

Because hold detection relies on key-repeat, avoid binding a bare letter key like `v` since it types into the prompt during warmup. Use `Space`, or use a modifier combination like `meta+k` to start recording on the first keypress with no warmup. See [customize keyboard shortcuts](/en/keybindings) for the full keybinding syntax.

## Troubleshooting

Common issues when voice dictation does not activate or record:

* **`Voice mode requires a Claude.ai account`**: you are authenticated with an API key or a third-party provider. Run `/login` to sign in with a Claude.ai account.
* **`Microphone access is denied`**: grant microphone permission to your terminal in system settings. On macOS, go to System Settings → Privacy & Security → Microphone. On Windows, go to Settings → Privacy → Microphone. Then run `/voice` again.
* **`No audio recording tool found` on Linux**: the native audio module could not load and no fallback is installed. Install SoX with the command shown in the error message, for example `sudo apt-get install sox`.
* **Nothing happens when holding `Space`**: watch the prompt input while you hold. If spaces keep accumulating, voice dictation is off; run `/voice` to enable it. If only one or two spaces appear and then nothing, voice dictation is on but hold detection is not triggering. Hold detection requires your terminal to send key-repeat events, so it cannot detect a held key if key-repeat is disabled at the OS level.
* **Transcription is garbled or in the wrong language**: dictation defaults to English. If you are dictating in another language, set it in `/config` first. See [Change the dictation language](#change-the-dictation-language).

## See also

* [Customize keyboard shortcuts](/en/keybindings): rebind `voice:pushToTalk` and other CLI keyboard actions
* [Configure settings](/en/settings): full reference for `voiceEnabled`, `language`, and other settings keys
* [Interactive mode](/en/interactive-mode): keyboard shortcuts, input modes, and session controls
* [Built-in commands](/en/commands): reference for `/voice`, `/config`, and all other commands

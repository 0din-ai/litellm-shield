# VHS documentation
#
# Output:
#   Output <path>.gif               Create a GIF output at the given <path>
#   Output <path>.mp4               Create an MP4 output at the given <path>
#   Output <path>.webm              Create a WebM output at the given <path>
#
# Require:
#   Require <string>                Ensure a program is on the $PATH to proceed
#
# Settings:
#   Set FontSize <number>           Set the font size of the terminal
#   Set FontFamily <string>         Set the font family of the terminal
#   Set Height <number>             Set the height of the terminal
#   Set Width <number>              Set the width of the terminal
#   Set LetterSpacing <float>       Set the font letter spacing (tracking)
#   Set LineHeight <float>          Set the font line height
#   Set LoopOffset <float>%         Set the starting frame offset for the GIF loop
#   Set Theme <json|string>         Set the theme of the terminal
#   Set Padding <number>            Set the padding of the terminal
#   Set Framerate <number>          Set the framerate of the recording
#   Set PlaybackSpeed <float>       Set the playback speed of the recording
#   Set MarginFill <file|#000000>   Set the file or color the margin will be filled with.
#   Set Margin <number>             Set the size of the margin. Has no effect if MarginFill isn't set.
#   Set BorderRadius <number>       Set terminal border radius, in pixels.
#   Set WindowBar <string>          Set window bar type. (one of: Rings, RingsRight, Colorful, ColorfulRight)
#   Set WindowBarSize <number>      Set window bar size, in pixels. Default is 40.
#   Set TypingSpeed <time>          Set the typing speed of the terminal. Default is 50ms.
#
# Sleep:
#   Sleep <time>                    Sleep for a set amount of <time> in seconds
#
# Type:
#   Type[@<time>] "<characters>"    Type <characters> into the terminal with a
#                                   <time> delay between each character
#
# Keys:
#   Escape[@<time>] [number]        Press the Escape key
#   Backspace[@<time>] [number]     Press the Backspace key
#   Delete[@<time>] [number]        Press the Delete key
#   Insert[@<time>] [number]        Press the Insert key
#   Down[@<time>] [number]          Press the Down key
#   Enter[@<time>] [number]         Press the Enter key
#   Space[@<time>] [number]         Press the Space key
#   Tab[@<time>] [number]           Press the Tab key
#   Left[@<time>] [number]          Press the Left Arrow key
#   Right[@<time>] [number]         Press the Right Arrow key
#   Up[@<time>] [number]            Press the Up Arrow key
#   Down[@<time>] [number]          Press the Down Arrow key
#   PageUp[@<time>] [number]        Press the Page Up key
#   PageDown[@<time>] [number]      Press the Page Down key
#   Ctrl+<key>                      Press the Control key + <key> (e.g. Ctrl+C)
#
# Display:
#   Hide                            Hide the subsequent commands from the output
#   Show                            Show the subsequent commands in the output
#   ScrollUp[@<time>] [number]      Scroll terminal viewport up by rows
#   ScrollDown[@<time>] [number]    Scroll terminal viewport down by rows

Output docs/demo-flag-adversarial.gif

Require bash
Require docker
Require curl
Require jq

Set Shell "bash"
Set FontSize 22
Set Width 1400
Set Height 800
Set Theme "Dracula"

# This is one of four single-act tapes (docs/demo-*.tape) that replaced the
# original all-in-one demo.tape. Each is independently self-contained: it does
# its own preflight, brings the proxy up in the mode its act needs, shows only
# that one call, then restores the proxy to flag mode before ending — so any
# of the four can be run standalone, in any order, repeatedly, without
# depending on another tape having run first or leaving particular state
# behind. scripts/demo_step.sh remains the single source of truth for every
# beat; only the choreography around it (what's Hidden vs Shown, timeouts,
# sentinels) lives here.
#
# Setup (preflight + up) and teardown (restore) are entirely Hidden — the user
# does not need to see preflight checks or a mode-switch narration to
# understand this act's own call. Whether VHS's Wait+Screen can actually be
# trusted to match a sentinel while Hide is still active — with no Show in
# between — was not documented anywhere and was verified empirically here
# (throwaway tape, outside this repo) before relying on it for the whole
# hidden setup/teardown below:
#
#   - A single Hidden Type→Enter→Wait+Screen cycle matches its sentinel and
#     returns as soon as it appears, exactly like a Shown one.
#   - Chaining multiple such cycles back to back under one Hide (no Show
#     between them) works the same way — each Wait+Screen still matches only
#     its own step's sentinel, not a leftover from the previous hidden step.
#   - Critically, a fully Hidden stretch contributes ZERO frames to the output
#     GIF (confirmed by comparing gif duration to the sum of only the Shown
#     Sleep/Type/Wait durations) — Hide is not "recorded but not decoded", it
#     is "not captured", so no workaround (Show+immediately-Hide-again, etc.)
#     is needed to get a silent setup phase.
#
# So the pattern below — Hide, then Type/Enter/Wait+Screen for preflight and
# up in sequence with no Show between them — is exactly as reliable as the
# Hidden/Shown pattern already validated in the original demo.tape for
# individual beats; it was just never exercised chained before.
#
# The clear before every typed command is still load-bearing for the same
# reason as in the original demo.tape: VHS's Wait+Screen gets permanently
# stuck once cumulative on-screen output needs to scroll past the visible
# terminal height, so each step's output must start from a blank screen.
#
# Folding `clear` into the same Hidden command line (`clear && scripts/...`)
# rather than typing it separately and Show-ing after avoids leaking the
# invocation line itself: the frame captured the instant Show resumes already
# reflects whatever sits in the screen buffer, and typing the invocation alone
# under Show would put it there before the real command's own output exists.
Hide
Type "clear && scripts/demo_step.sh preflight" Enter
Wait+Screen@20s /Preflight checks passed/
Type "clear && scripts/demo_step.sh up flag" Enter
Wait+Screen@180s /Proxy ready in flag mode/
Show

# The only visible beat: an adversarial prompt in flag (monitor-only) mode —
# HTTP 200, annotated with a high suspicion score, but NOT blocked. Scored in
# parallel with the live model call (during_call), so the score costs ~0
# perceived latency on screen.
Hide
Type "clear && scripts/demo_step.sh call flag-adversarial" Enter
Show
Wait+Screen@90s /flag-adversarial prompt: HTTP/
Sleep 4s

# Teardown — always leave the proxy resting in flag mode, hidden, so this tape
# is safe to re-run standalone regardless of what ran before it.
Hide
Type "clear && scripts/demo_step.sh restore" Enter
Wait+Screen@180s /restored to flag mode/

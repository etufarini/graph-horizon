<!--
This index owns navigation for the installed command-line interface. Runtime
options, terminal behavior, and local slash commands remain separate contracts.
-->

# Command-Line Interface

The `graph-horizon` executable starts the interactive terminal by default or
the integrated Web UI when `--mode web` is selected.

```sh
graph-horizon --model /absolute/path/to/model.gguf
```

Use the page matching the information you need:

| Document | Contents |
|---|---|
| [Runtime options](runtime-options.md) | Every accepted `graph-horizon` flag, value, default, and mode restriction |
| [Terminal interface](terminal-interface.md) | Keyboard controls, streaming, rendering, and status information |
| [Slash commands and file attachments](slash-commands-and-file-attachments.md) | `/clear`, `/system`, `/import`, `/export`, completion, and `@file` behavior |

Installer flags are not runtime options. They are documented in the
[installation guide](../installation.md).

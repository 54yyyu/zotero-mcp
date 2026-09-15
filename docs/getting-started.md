# Getting Started with Zotero MCP

This guide walks you through installing Zotero MCP and connecting it to your AI assistant. For every setting, see [Configuration](configuration.md); for problems, see [Troubleshooting](troubleshooting.md).

**Requirements**
- Python 3.10+
- Zotero 7+ (for local API with full-text access)
- An MCP-compatible client (e.g., Claude Desktop, ChatGPT Developer Mode, Cherry Studio, Chorus)

## Installation

```bash
uv tool install zotero-mcp-server   # recommended
pip install zotero-mcp-server       # or with pip
pipx install zotero-mcp-server      # or with pipx
```

Optional extras (`semantic`, `pdf`, `scite`, `all`) are listed in the [README](../README.md#optional-extras). If you are new to the command line, the community-built [Zotero MCP Setup](https://github.com/ehawkin/zotero-mcp-setup) has a macOS GUI installer (DMG), one-click install scripts for Mac and Windows, and a step-by-step guide.

## Configure Zotero

The server needs to know how to connect to your Zotero library. There are two main ways to do this.

### Option 1: Local Zotero (Recommended)

If you're running Zotero 7 or later on the same machine, you can connect to the local API:

1. Allow local connections in Zotero's settings:
   - Open Zotero
   - Open Settings (Edit → Settings on Windows/Linux, Zotero → Settings on macOS) → Advanced → Miscellaneous
   - Tick "Allow other applications on this computer to communicate with Zotero"

   ![Zotero local API](zotero-local-api.png)

2. Set the environment variable:
   ```bash
   export ZOTERO_LOCAL=true
   ```

For **writes** you have two routes. On Zotero 10 or newer, run `zotero-mcp authorize-local` once and writes go straight to the running Zotero — see [Local write support](configuration.md#local-write-support). On any older Zotero the local API is read-only, so also set the web API variables below and the server writes through the Zotero web API instead ("hybrid mode": fast local reads, web API writes).

### Option 2: Zotero Web API

If you want to connect to your Zotero library via the web API:

1. Get your Zotero API key:
   - Go to [https://www.zotero.org/settings/keys](https://www.zotero.org/settings/keys)
   - Create a new key with appropriate permissions (at least "Read" access)

2. Find your library ID:
   - For personal libraries, your user ID is available at the same page
   - For group libraries, it's the number in the URL when viewing the group

3. Set the environment variables:
   ```bash
   export ZOTERO_API_KEY=your_api_key
   export ZOTERO_LIBRARY_ID=your_library_id
   export ZOTERO_LIBRARY_TYPE=user  # or 'group' for group libraries
   ```

## Integrating with Claude Desktop and Claude Code

1. **Auto-configure** (recommended):
   ```bash
   zotero-mcp setup
   ```

   `zotero-mcp setup` probes every known location of `claude_desktop_config.json`, writes to each one it finds, and prints the absolute path(s) it wrote so you can confirm it matched the build you actually run.

2. **Manual configuration**: for Claude Desktop, open its configuration file:
   - On macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - On Windows: `%APPDATA%\Claude\claude_desktop_config.json`

   Some Claude Desktop builds store the file elsewhere, for example
   `%LOCALAPPDATA%\Claude-3p\claude_desktop_config.json` on Windows or
   `~/Library/Application Support/Claude-3p/claude_desktop_config.json` on macOS.

   For Claude Code, add the server to `~/.claude.json`. The entry is the same for both:
   ```json
   {
     "mcpServers": {
       "zotero": {
         "command": "zotero-mcp",
         "env": {
           "ZOTERO_LOCAL": "true",
           "ZOTERO_API_KEY": "YOUR_API_KEY",
           "ZOTERO_LIBRARY_ID": "YOUR_LIBRARY_ID"
         }
       }
     }
   }
   ```

   For **local reads**, `ZOTERO_LOCAL: "true"` is all you need — drop the `ZOTERO_API_KEY` and `ZOTERO_LIBRARY_ID` lines entirely. Keep them only for web API writes on a Zotero older than 10 (for a group library, also set `ZOTERO_LIBRARY_TYPE: "group"`).

   > **Important Note**: Environment variables set in the shell you run `claude` in will override these values.

   > **Tip:** If Claude Desktop reports it can't find the `zotero-mcp` command, use the
   > absolute path instead (run `zotero-mcp setup-info` or `which zotero-mcp` to find it) —
   > GUI apps don't always inherit your shell `PATH`.

3. **Use it**:
   1. Start Zotero desktop (make sure the local API is enabled)
   2. Launch Claude Desktop / Claude Code
   3. In Claude Desktop, the Zotero tools appear in the tools interface (if not, check the connections menu under Settings). In Claude Code, run `/mcp` and make sure the Zotero server is connected.

If your agent has a shell (Claude Code, Cursor, Codex …), you can use `zotero-cli` through an agent skill instead of the MCP server, at a fraction of the context cost: `zotero-mcp install-skill`. See [CLI and agent skill](cli.md).

## Integrating with OpenAI's ChatGPT

This option is available through the ChatGPT web app. You must use [ChatGPT Developer mode](https://platform.openai.com/docs/guides/developer-mode) which may be restricted to a limited number of OpenAI platforms and apps. A paid subscription appears to be required.

zotero-mcp is not available by default as a web-based MCP, and it seems likely that many users will want to stick with a local MCP due to their large document libraries. Since ChatGPT does not support local MCPs natively through their desktop app (yet?) the way you can move forward is by tunneling.

**Use at your own risk**

`zotero-mcp serve` has no authentication of its own. Treat the tunnel URL as a bearer token: anyone who has it can use every tool with whatever access the running server has, including writes when web API credentials are configured. The connector recipe below sends no credentials, so ngrok's basic auth would block it; what protects you is keeping the URL private, stopping the tunnel whenever you are not using it, and any ngrok traffic policy (IP or method restrictions) you can apply. If your ChatGPT account offers OAuth for connectors, prefer it over `No authentication`. Whatever the AI service can read from your library is exposed to that service; judge that for your own situation before continuing.

### Setting up a desktop tunnel for zotero-mcp

A tunnel makes your locally running `zotero-mcp` server securely available to a web service like ChatGPT. We recommend [ngrok](https://ngrok.com/) for this.

1.  **Install ngrok**: Follow the instructions on the [ngrok website](https://ngrok.com/download) to download and install it. Mac users can use `brew` and we have successfully tested this approach.

2.  **Start the `zotero-mcp` server**: Before starting the tunnel, make sure your MCP server is running. For web-based clients use the `streamable-http` transport (`sse` still works but is deprecated and prints a warning). Open a terminal and run:
    ```bash
    # Make sure your Zotero environment variables are set first!
    # e.g., export ZOTERO_LOCAL=true
    zotero-mcp serve --transport streamable-http --port 8000
    ```
    Leave `--host` at its default (localhost): ngrok forwards to localhost, and binding `0.0.0.0` would also expose the server to your local network.

Important: you should probably leave this terminal open in order to ensure tunnel traffic is successfully transiting to the server.

3.  **Start the ngrok tunnel**: Open a *second* terminal and start ngrok, pointing it to the port your server is using (8000). Here is an instruction that will work on a mac
    ```bash
    ngrok http 8000
    ```
4.  **Copy the URL**: Ngrok will provide a public `Forwarding` URL that looks something like `https://<random-string>.ngrok-free.app`. Copy this HTTPS URL—you'll need it for the ChatGPT connector setup.

### Setting up a ChatGPT or OpenAI client for zotero-mcp
There are actually two ways to work with ChatGPT on the web once you have a tunnel open to your server: through the ChatGPT app at [chatgpt.com](https://chatgpt.com), or through the chat prompt builder screen at the [OpenAI platform page](https://platform.openai.com/chat).

The setup is nearly identical for both.

#### 1. ChatGPT.com setup

1.  Navigate to [chatgpt.com](https://chatgpt.com). Make sure you are logged in, and at the base "chat" user interface.
2.  Click on your profile name, then **Settings**.
3.  Go to the **Connectors** tab:
    *   First you must enable "Developer Mode." At the bottom of the connectors tab, there is an "Advanced..." button. Click this and then on the next screen enable "Developer Mode."
    *   Now from the main Connectors browser window, click **Create**
4.  Fill in the details:
    *   **Name**: Zotero MCP
    *   **Description**: Search and retrieve documents from a local Zotero library.
    *   **MCP Server URL**: This is the critical part. With the `streamable-http` transport the endpoint is `/mcp` (example: `https://<YOUR_NGROK_URL>.ngrok-free.app/mcp`). The steps below were last verified with the deprecated `sse` transport; if the connector does not accept the `/mcp` URL, start the server with `--transport sse` and use the `/sse/` form: combine your ngrok URL, the `/sse/` endpoint (with a trailing slash), and a unique `session_id`.
        *   If you use the `sse` fallback: the trailing slash on `/sse/` is important to avoid a redirect.
        *   The `session_id` must be a valid [UUIDv4](https://www.uuidgenerator.net/). While some clients might negotiate a session automatically, explicitly providing a unique ID is the most reliable method.
        *   Example URL: `https://<YOUR_NGROK_URL>.ngrok-free.app/sse/?session_id=<YOUR_UUID>`
    *   **Authentication**: `No authentication` — the server has none, and this recipe sends no credentials; see "Use at your own risk" above.
    *   Tick the "I trust this application" checkbox.
5.  Click **Create**. If you are successfully connecting you should see relevant communications logs in your tunnel and your server terminals. If this is successful, an important indication will be the listing of all zotero-mcp tools in the ChatGPT interface.
    *   *Important: our testing indicates that you need to turn all the "Edit" sliders to "Off" in the list of tools.* Otherwise the tool may not be enabled in Developer Mode.

      ![ChatGPT Connector Tool List](../public/ChatGPT_zot_mcp_1.png)

6.  You should now be ready to add Zotero-MCP to new chats. To do this, go to the main ChatGPT interface. It should indicate that you are in development mode. When you start a new chat, click the "plus" icon in the text box interface to select "Deep Research" as a chat mode. A "Sources" menu will become available: enable Zotero-MCP as one of the sources:

      ![Enable Zotero-MCP as a Source](../public/ChatGPT_zot_mcp_2.png)

#### 2. OpenAI Chat Builder setup

The process is the same as above, but you create the connector within the context of building a custom GPT on the OpenAI Platform.

1.  Navigate to the [OpenAI Platform Chat page](https://platform.openai.com/chat).
2.  When configuring a custom GPT, go to the **Tools** section and choose to add an MCP connector.
3.  Follow the same steps as in the `ChatGPT.com setup` to configure the connector URL and other details.

## Integrating with Cherry Studio

Go to Settings -> MCP Servers -> Edit MCP Configuration, and add the following:

```json
{
  "mcpServers": {
    "zotero": {
      "name": "zotero",
      "type": "stdio",
      "isActive": true,
      "command": "zotero-mcp",
      "args": [],
      "env": {
        "ZOTERO_LOCAL": "true"
      }
    }
  }
}
```

Then click "Save". Cherry Studio also provides a visual configuration method for general settings and tools selection.

## Integrating with Chorus.sh

[Chorus.sh](https://chorus.sh) is a popular multi-chatbot interface that configures MCP servers through an online preferences form rather than config files.
This would be one possible path to working with Zotero with chatbots other than Claude.

To set up Zotero MCP with Chorus.sh:

1. **Find your installation path**:
   - For `uv tool install`: `~/.local/bin/zotero-mcp` on macOS and Linux
   - For other methods: use `zotero-mcp setup-info` to get the exact path and configuration details

2. **Configure in Chorus.sh preferences**:
   - **Command**: Enter the full path to your zotero-mcp installation
   - **Arguments**: Leave empty (no custom --port or --host arguments needed unless set at config time)
   - **Environment (JSON)**: Take your environment configuration JSON (including outer brackets), remove newlines, and paste as a single line

3. **Example Environment JSON** (single line format):
   ```json
   {"ZOTERO_LOCAL": "true"}
   ```

Many other MCP consumers use similar configuration approaches with command path, arguments, and environment variables.

## Integrating with Autohand Code

After installing Zotero MCP, add a local read-only server with:

```bash
autohand mcp add zotero env ZOTERO_LOCAL=true zotero-mcp
```

Add `--scope project` after `add` to keep the server configuration in the current project. For hybrid or web API access, add the credentials described above to the `env` command. See [Autohand Code](https://github.com/autohandai/code-cli/) for current installation and CLI details.

## Using with Other MCP Clients

Zotero MCP works with any MCP-compatible client. You can start the server manually:

```bash
zotero-mcp serve --transport stdio
```

For HTTP-based clients:

```bash
zotero-mcp serve --transport streamable-http --host localhost --port 8000
```

The `sse` transport is still accepted but deprecated.

## Available Tools

The full, current tool list is in [Tools](tools.md). Search, metadata, full text, collections, tags, notes, annotations, PDF reading, adding and editing items, and semantic search are all covered; some groups are opt-in via `ZOTERO_MCP_TOOLSETS`.

## Example Queries

Once connected, you can ask things like:

- "Search my library for papers on machine learning"
- "Find recent articles I've added about climate change"
- "Summarize the key findings from my paper on quantum computing"
- "Extract all PDF annotations from my paper on neural networks"
- "Search my notes and annotations for mentions of 'reinforcement learning'"
- "Show me papers tagged '#Arm' excluding those with '#Crypt' in my library"
- "Export the BibTeX citation for papers on machine learning"
- "Highlight the main claims in this paper and box its key figures"
- **"Find papers conceptually similar to deep learning in computer vision"** *(semantic search)*
- **"Papers that discuss topics similar to this abstract: [paste text]"** *(semantic search)*

Something not working? See [Troubleshooting](troubleshooting.md).

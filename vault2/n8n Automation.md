---
tags: [automation, n8n, workflows, self-hosted]
type: tool
status: running
url: https://hanishchow.app.n8n.cloud
related: [[Obsidian Setup]]
---

# ⚡ n8n Automation

> [!abstract] One Line
> Self-hosted workflow automation. My personal automation layer connecting apps without writing full applications.

**URL:** https://hanishchow.app.n8n.cloud

---

## Connected Services

| Service | Status | Connected Via |
|---------|--------|--------------|
| Google Calendar | ✅ Active | MCP |
| Gmail | ✅ Active | MCP |
| Claude (AI steps) | ✅ Active | Anthropic API |

---

## Core Concepts

### Node Types
```
[Trigger]  →  [Transform]  →  [Action]
   ↑               ↑              ↑
starts flow    modifies data    does something
```

**Triggers:** webhook, schedule, new email, new event
**Transforms:** Set, Function, Merge, Filter
**Actions:** send email, create event, HTTP request, AI call

---

## Example: AI Email Summariser

```
[Gmail: New Email]
    → [Filter: has attachment]
    → [HTTP: Anthropic API — summarise]
    → [Gmail: Send summary to me]
```

---

## Planned Workflows

- [ ] Auto-summarise research papers from Gmail
- [ ] PDBBind / arxiv new paper notifier
- [ ] Obsidian daily note from calendar events
- [ ] Molecule result logger → Google Sheets

---

## References
- [n8n Docs](https://docs.n8n.io)
- [n8n GitHub](https://github.com/n8n-io/n8n)

---
tags: [obsidian, second-brain, setup, meta, pkm]
type: setup
status: active
related: [[About Me]]
---

# 🗂️ Obsidian Setup

> [!abstract] One Line
> My second brain. Everything I know lives here as an interconnected graph. Links > folders.

---

## Philosophy

> [!quote] Core Belief
> Knowledge is a **graph**, not a tree. Folders are a lie. Links are the truth.

- Every concept = one note
- Every note links to related notes
- The graph view IS the knowledge map
- Update notes as you learn, not after

---

## Vault Folder Structure

```
vault/
├── 📁 00 - Core/
│   └── About Me.md              ← root node
├── 📁 01 - Projects/
│   └── Binding Affinity Prediction Model.md
├── 📁 02 - Tools/
│   ├── AutoDock Vina.md
│   ├── DiffDock.md
│   ├── GNINA.md
│   ├── RDKit.md
│   ├── ProLIF.md
│   ├── oddt.md
│   ├── MDAnalysis.md
│   └── n8n Automation.md
├── 📁 03 - Concepts/
│   ├── Vinardo.md
│   ├── PLEC Fingerprints.md
│   └── vina_score.py.md
├── 📁 04 - Datasets/
│   └── PDBBind.md
├── 📁 05 - Setup/
│   ├── WSL Setup.md
│   ├── AMD GPU ROCm.md
│   ├── Tools Installed.md
│   └── Obsidian Setup.md
├── 📁 06 - Papers/
│   └── Papers to Read.md
└── 📁 07 - Code/
    └── GEOCK.md
```

---

## Frontmatter Standard

> [!tip] Use This on Every Note
> ```yaml
> ---
> tags: [relevant, tags]
> type: tool / concept / project / dataset / setup / paper / code
> status: active / installed / mastered / in-progress / skip / todo
> related: [[Link1]], [[Link2]]
> created: YYYY-MM-DD
> ---
> ```

---

## Formatting Cheatsheet

| Feature | Syntax | Use for |
|---------|--------|---------|
| **Bold** | `**text**` | key terms |
| ==Highlight== | `==text==` | critical info |
| ~~Strikethrough~~ | `~~text~~` | deprecated/wrong |
| `inline code` | `` `code` `` | commands, values |
| Wikilink | `[[Note Name]]` | linking concepts |
| Callout | `> [!type] Title` | warnings, tips, bugs |
| Heading | `## H2`, `### H3` | sections |
| Math inline | `$formula$` | equations |
| Math block | `$$formula$$` | big equations |

### Callout Types

```markdown
> [!abstract]   one-line summary
> [!note]        neutral info
> [!tip]         good to know
> [!important]   critical insight
> [!warning]     watch out
> [!caution]     potential issue
> [!error]       broken / blocked
> [!bug]         known bug
> [!check]       verified / done
> [!todo]        action needed
> [!quote]       philosophy / motto
```

---

## Essential Plugins

| Plugin | Purpose |
|--------|---------|
| **Graph View** | Visualise knowledge connections |
| **Backlinks** | See what links to current note |
| **Dataview** | Query notes like a database |
| **Templater** | Auto-fill frontmatter on new notes |
| **Calendar** | Daily notes linked to dates |
| **Tag Wrangler** | Manage tags across entire vault |

---

## Key Shortcuts

| Action | Shortcut |
|--------|---------|
| New note | `Ctrl+N` |
| Follow link | `Ctrl+Click` |
| Graph view | `Ctrl+G` |
| Search vault | `Ctrl+Shift+F` |
| Command palette | `Ctrl+P` |
| Create wikilink | type `[[` |
| Toggle preview | `Ctrl+E` |

---

## The Workflow

> [!important] Every Time I Learn Something New
> 1. Create a note with frontmatter
> 2. Link it to related notes with `[[wikilinks]]`
> 3. Update `[[About Me]]` if it's a new skill
> 4. Add to relevant project note if applicable
> 5. Graph grows → brain grows

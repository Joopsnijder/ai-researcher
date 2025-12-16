---
marp: true
theme: default
paginate: true
backgroundColor: #fff
html: true
style: |
  section {
    font-size: 28px;
  }
  section > div > img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 20px auto;
  }
  p img {
    display: inline !important;
    vertical-align: middle;
    margin: 0 0.2em;
  }
---

# My Presentation Title
## Subtitle

---

## Introduction

This is a sample slide with a Mermaid diagram.

---

## Diagram Example

```mermaid
graph TD
    A[Start] --> B[Process]
    B --> C{Decision}
    C -->|Yes| D[Success]
    C -->|No| E[Retry]
    E --> B
```

---

## Features

- ✅ Mermaid diagrams automatically converted to SVG
- ✅ Responsive scaling
- ✅ Emoji support
- 📊 Easy to maintain

---

## Next Steps

1. Edit this markdown file
2. Add your content and Mermaid diagrams
3. Run `npm run build:presentations docs/your-file.md`
4. Open the generated HTML or PDF

---

# Thank You! 🎉

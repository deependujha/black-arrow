# Black Arrow

An mdBook of notes on ML performance engineering. Pages live in `src/` and are listed in `src/SUMMARY.md`.

## Writing notes

- **Never hotlink images.** Download every image into the `images/` folder next to the page (for example `src/00_llm_core/images/`) and reference it with a relative path, so links cannot break.
- Give each image a descriptive `snake_case` filename and meaningful alt text.
- Embed with `<img src="images/name.png" alt="..." style="background: #fff; padding: 8px; border-radius: 6px;">` so it stays readable in dark themes.
- Put a credit line directly under each image: `*Image: [source](url) (license).*`. This is personal & private learning resource, so strict license compliance is not required.
- Notes should be easy to follow, well structured, and end to end. Do not repeat information, but when in doubt prefer a longer page over leaving something out.
- Math renders with MathJax (`mathjax-support = true` in `book.toml`). Inline: `\\( x^2 \\)`. Display: `\\[ x^2 \\]` or `$$ x^2 $$`. Single `$...$` does not work. Keep shape traces and ASCII pipelines in `text` code blocks.

## Build

`mdbook build` must pass before finishing. `create-missing = false`, so every page in `SUMMARY.md` must exist.

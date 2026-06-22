# Git Hooks

Shared hooks for this repository. Enable once after cloning:

```bash
git config user.name "John Bee"
git config user.email "jnbee2010@gmail.com"
git config core.hooksPath githooks
```

## prepare-commit-msg

Removes `Co-authored-by` trailers that reference Cursor or cursoragent from commit messages, so commits stay attributed to John Bee only.

---
description: Move a completed ticket from .tickets to .complete and push changes to GitHub
---

// turbo-all
1. Identify the ticket file in `.tickets` that has been completed.
2. Move the completed ticket from `.tickets` to `.complete`.
   ```bash
   mv .tickets/<ticket-filename> .complete/
   ```
3. Stage all changes, including the moved ticket and the implementation work.
   ```bash
   git add .
   ```
4. Commit the changes with a descriptive message.
   ```bash
   git commit -m "Complete ticket: <ticket-summary>"
   ```
5. Push the changes to the current GitHub branch.
   ```bash
   git push origin $(git rev-parse --abbrev-ref HEAD)
   ```

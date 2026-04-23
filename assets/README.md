# Assets

Place optional external assets here.

- `maps/`: occupancy map images (PNG preferred).
  - White/light pixels = drivable space
  - Black/dark pixels = walls/obstacles
- `sprites/`: car sprite images (PNG with alpha channel preferred)

If no compatible map or sprite exists, the visualization system falls back to fully procedural generation.

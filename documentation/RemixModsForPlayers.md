# Installing Remix mods

A Remix mod replaces game materials, meshes and lights with path-traced ones. It lives in the game folder, in `rtx-remix\mods\<ModName>\` (the folder holds a `mod.usda`, `mod.usd` or `mod.usdc` and the mod's other files). Every mod there is loaded.

## One order: your mod manager's

When two Remix mods replace the same thing, the one your mod manager puts last wins, the same way it decides between any two mods that contain the same file:

| How you install | The winner |
|-----------------|-----------|
| Mod Organizer 2 with Root Builder | The mod lower in the left pane (higher priority). |
| Vortex | The mod that loads after the other. Vortex asks for a rule when two mods share files. |
| By hand | The archive you extracted last, if you let it overwrite files. |

This works because each mod ships a small file per thing it replaces, in `rtx-remix\mods\owners\`. Two mods that replace the same thing ship the same file, and your mod manager picks which copy stays. The game follows that copy. Mods without these files are ordered by folder name (a later name wins), and a mod that lists another as a dependency wins over it.

## Mod Organizer 2 (Root Builder)

Remix mods go in the game folder itself, not in `Data`, so MO2 needs the [Root Builder](https://www.nexusmods.com/site/mods/31) plugin.

1. Install the archive in MO2 as usual.
2. If the archive has `rtx-remix` at its top level, open the mod's folder (right-click > Open in Explorer), make a folder named `Root`, and move `rtx-remix` into it. Some mods offer an MO2 download that already has `Root\rtx-remix\...`.
3. Enable the mod and order it in the left pane. When two Remix mods conflict, MO2 shows the lightning icons, and the lower one wins.
4. Launch through MO2 as usual. Root Builder places the files in the game folder.

## Vortex

1. Install the archive. Its `rtx-remix` folder must go to the game folder, not to `Data`: if Vortex placed it in `Data`, set the mod's type to one that deploys to the game folder (right-click the mod > Set Mod Type > **Engine Injector**, or **dinput** where Vortex offers that instead) and deploy again.
2. Deploy. When Vortex reports a file conflict between two Remix mods, choose which loads after: that one wins.

## By hand

Extract the archive into the game folder, so that its `rtx-remix` folder merges with the one already there. When Windows asks about replacing files in `rtx-remix\mods\owners`, **Replace** makes the mod you are extracting win the conflicts; **Skip** keeps the earlier one.

To remove a mod, delete its folder in `rtx-remix\mods`. The `owners` files it leaves behind are ignored.

## Checking the result

- The Remix menu's **Enhancements > Mod Conflicts** lists your Remix mods from lowest to highest priority, how much each replaces, and every conflict with the winner and why it won (ownership file, dependency, name or load order).
- `rtx-remix\logs\remix-dxvk.log` has the same as `[ModOrder]` lines.
- After you change the order with the game running, press **Reload Enhancements** in the Remix menu or restart the game.

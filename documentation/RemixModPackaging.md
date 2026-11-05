# Packaging a Remix mod

How to package a Remix mod so that players' mod managers decide what happens when it and another mod replace the same asset. How the runtime uses the result: [RemixModOrder.md](RemixModOrder.md). What players do: [RemixModsForPlayers.md](RemixModsForPlayers.md).

## Steps

1. **Export from the Remix Toolkit** (Mod Packaging). Make sure the mods yours builds on are its dependencies: the Toolkit writes them into `lightspeed_mod_dependencies` in `mod.usda`, and the runtime ranks your mod above them, so your changes win where you override theirs. A dependency may name the other mod's folder, a path to it or to its `mod.usda`, or its `lightspeed_mod_name`.
2. **Lay out the folder** the way it goes into the game folder:
   ```
   rtx-remix\
     mods\
       MyMod\
         mod.usda
         (sublayers, textures, meshes...)
   ```
   The folder name (`MyMod`) is your mod's identity. Keep it the same in every version: ownership files and other mods' dependency lists name it.
3. **Write the ownership files** with `remix_mod_owners` from that folder's parent:
   ```
   remix_mod_owners.exe rtx-remix\mods\MyMod
   ```
   It reads `mod.usda` with its sublayers, finds every material, mesh and light the mod replaces (`mat_<HASH>`, `mesh_<HASH>`, `light_<HASH>` under `/RootNode`), and writes one file per asset to `rtx-remix\mods\owners\`, each holding `MyMod`. Run it again after every export. Add `--prune` to delete the files of assets you no longer replace.
4. **Archive** the `rtx-remix` folder, `owners` included:
   ```
   rtx-remix\mods\MyMod\...
   rtx-remix\mods\owners\mat_0123456789ABCDEF
   rtx-remix\mods\owners\mesh_...
   ```
   The folder is `owners`, with no leading dot: Mod Organizer 2's Root Builder does not copy names that start with a dot.
   This layout installs by hand and with Vortex. For Mod Organizer 2 with Root Builder, offer a second archive with the same content under `Root\` (`Root\rtx-remix\mods\...`), or tell players to move it there.

## Notes

- Run the tool in your packaging folder, not in a game folder with other Remix mods installed: it overwrites the ownership files it finds, other mods' included. In a game folder, `--check` lists which of your assets other installed mods also replace without changing anything.
- Ship only the ownership files the tool writes for your mod. A file for an asset your mod does not replace is ignored, and one naming another mod makes yours lose that asset.
- A material's hash is the game texture's hash, the one the Remix menu's texture picker shows. For games drawn through the Remix API it is matched as well, as long as the game imports its own textures.
- `remix_mod_owners.exe` runs on the CPU and needs the USD DLLs and the `usd` plugin folder beside it, as `tools\remix_mod_owners\build.bat` lays them out in `bin\`.

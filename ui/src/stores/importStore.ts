import { create } from "zustand";
import type {
  Place,
  FolderNode,
  ImportFile,
  SortField,
  SortDir,
} from "../types/import";
import { listFolders, listFiles, getHomePath, importImages, copyAndImportImages } from "../api/commands";
import { emit } from "../events/eventBus";

export type ImportMode = "inplace" | "copy";

interface ImportState {
  isOpen: boolean;
  importMode: ImportMode;

  // Places
  places: Place[];
  selectedPlacePath: string | null;

  // Folders
  folderTree: FolderNode[];
  selectedFolderPath: string | null;

  // Files
  files: ImportFile[];

  // Options
  selectOnlyNew: boolean;
  recursive: boolean;
  ignoreNonRaw: boolean;

  // Sort
  sortField: SortField;
  sortDir: SortDir;

  // Layout
  leftPanelWidth: number;

  // Actions
  openDialog: (mode?: ImportMode) => void;
  closeDialog: () => void;
  selectPlace: (path: string) => void;
  selectFolder: (path: string) => void;
  expandFolder: (path: string) => void;
  collapseFolder: (path: string) => void;
  toggleFileSelection: (fullpath: string, e?: { ctrlKey?: boolean; metaKey?: boolean; shiftKey?: boolean }) => void;
  selectAll: () => void;
  selectNone: () => void;
  selectNew: () => void;
  setSort: (field: SortField) => void;
  setSelectOnlyNew: (v: boolean) => void;
  setRecursive: (v: boolean) => void;
  setIgnoreNonRaw: (v: boolean) => void;
  addCustomPlace: (path: string) => void;
  removePlace: (path: string) => void;
  setLeftPanelWidth: (w: number) => void;
  doImport: () => void;
}

// Helper: update a node deep in the tree by path
function updateTree(
  nodes: FolderNode[],
  targetPath: string,
  updater: (node: FolderNode) => FolderNode,
): FolderNode[] {
  return nodes.map((node) => {
    if (node.path === targetPath) return updater(node);
    if (targetPath.startsWith(node.path + "/") && node.children) {
      return { ...node, children: updateTree(node.children, targetPath, updater) };
    }
    return node;
  });
}

function sortFiles(files: ImportFile[], field: SortField, dir: SortDir): ImportFile[] {
  const sorted = [...files];
  sorted.sort((a, b) => {
    let cmp: number;
    if (field === "name") {
      cmp = a.filename.localeCompare(b.filename);
    } else {
      cmp = a.modified - b.modified;
    }
    return dir === "asc" ? cmp : -cmp;
  });
  return sorted;
}

/** Fetch subfolders and convert to FolderNode[] */
async function fetchFolders(path: string): Promise<FolderNode[]> {
  try {
    const entries = await listFolders(path);
    console.log("[import] listFolders", path, "→", entries?.length, "entries");
    return entries
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((e) => ({
        name: e.name,
        path: e.path,
        children: e.hasChildren ? null : [],
        expanded: false,
      }));
  } catch (err) {
    console.error("[import] listFolders failed:", err);
    return [];
  }
}

/** Fetch files and convert to ImportFile[] */
async function fetchFiles(
  path: string,
  recursive: boolean,
  ignoreNonRaw: boolean,
  selectOnlyNew: boolean,
): Promise<ImportFile[]> {
  try {
    const entries = await listFiles(path, recursive, ignoreNonRaw);
    console.log("[import] listFiles", path, "→", entries?.length, "files");
    return entries.map((e) => ({
      ...e,
      selected: selectOnlyNew ? !e.alreadyImported : true,
    }));
  } catch (err) {
    console.error("[import] listFiles failed:", err);
    return [];
  }
}

/** Fetch default places using the C-side getHomePath binding */
async function getDefaultPlacesAsync(): Promise<Place[]> {
  try {
    const home = await getHomePath();
    console.log("[import] home path:", home);
    return [
      { name: "home", path: home, type: "home" },
      { name: "pictures", path: home + "/Pictures", type: "pictures" },
    ];
  } catch (err) {
    console.error("[import] getHomePath failed:", err);
    return [{ name: "/", path: "/", type: "home" }];
  }
}

export const useImportStore = create<ImportState>((set, get) => ({
  isOpen: false,
  importMode: "inplace",
  places: [],
  selectedPlacePath: null,
  folderTree: [],
  selectedFolderPath: null,
  files: [],
  selectOnlyNew: true,
  recursive: false,
  ignoreNonRaw: true,
  sortField: "modified",
  sortDir: "asc",
  leftPanelWidth: 220,

  openDialog: (mode = "inplace") => {
    set({
      isOpen: true,
      importMode: mode,
      places: [],
      selectedPlacePath: null,
      folderTree: [],
      selectedFolderPath: null,
      files: [],
    });
    getDefaultPlacesAsync().then((places) => {
      const defaultPlace = places.find((p) => p.type === "pictures") ?? places[0];
      set({ places, selectedPlacePath: defaultPlace.path });
      fetchFolders(defaultPlace.path).then((folderTree) => {
        if (get().selectedPlacePath === defaultPlace.path) set({ folderTree });
      });
    });
  },

  closeDialog: () => {
    set({
      isOpen: false,
      files: [],
      folderTree: [],
      selectedFolderPath: null,
    });
  },

  selectPlace: (path) => {
    set({
      selectedPlacePath: path,
      folderTree: [],
      selectedFolderPath: null,
      files: [],
    });
    fetchFolders(path).then((folderTree) => {
      if (get().selectedPlacePath === path) set({ folderTree });
    });
  },

  selectFolder: (path) => {
    const { recursive, ignoreNonRaw, sortField, sortDir, selectOnlyNew } = get();
    set({ selectedFolderPath: path, files: [] });
    fetchFiles(path, recursive, ignoreNonRaw, selectOnlyNew).then((files) => {
      if (get().selectedFolderPath === path) {
        set({ files: sortFiles(files, sortField, sortDir) });
      }
    });
  },

  expandFolder: (path) => {
    const { folderTree } = get();
    // Mark as expanded immediately (children stay null while loading)
    set({
      folderTree: updateTree(folderTree, path, (node) => ({
        ...node,
        expanded: true,
      })),
    });
    // If children haven't been loaded yet, fetch them
    const findNode = (nodes: FolderNode[]): FolderNode | null => {
      for (const n of nodes) {
        if (n.path === path) return n;
        if (n.children) {
          const found = findNode(n.children);
          if (found) return found;
        }
      }
      return null;
    };
    const node = findNode(folderTree);
    if (node && node.children === null) {
      fetchFolders(path).then((children) => {
        const currentTree = get().folderTree;
        set({
          folderTree: updateTree(currentTree, path, (n) => ({
            ...n,
            expanded: true,
            children,
          })),
        });
      });
    }
  },

  collapseFolder: (path) => {
    const { folderTree } = get();
    set({
      folderTree: updateTree(folderTree, path, (node) => ({
        ...node,
        expanded: false,
      })),
    });
  },

  toggleFileSelection: (fullpath, e) => {
    const multi = e?.ctrlKey || e?.metaKey;
    set((s) => ({
      files: s.files.map((f) => {
        if (multi) {
          return f.fullpath === fullpath ? { ...f, selected: !f.selected } : f;
        }
        return { ...f, selected: f.fullpath === fullpath };
      }),
    }));
  },

  selectAll: () => {
    set((s) => ({ files: s.files.map((f) => ({ ...f, selected: true })) }));
  },

  selectNone: () => {
    set((s) => ({ files: s.files.map((f) => ({ ...f, selected: false })) }));
  },

  selectNew: () => {
    set((s) => ({
      files: s.files.map((f) => ({ ...f, selected: !f.alreadyImported })),
    }));
  },

  setSort: (field) => {
    const { sortField, sortDir, files } = get();
    const newDir = field === sortField ? (sortDir === "asc" ? "desc" : "asc") : "asc";
    set({
      sortField: field,
      sortDir: newDir,
      files: sortFiles(files, field, newDir),
    });
  },

  setSelectOnlyNew: (v) => {
    set({ selectOnlyNew: v });
    if (v) get().selectNew();
  },

  setRecursive: (v) => {
    set({ recursive: v });
    const { selectedFolderPath } = get();
    if (selectedFolderPath) get().selectFolder(selectedFolderPath);
  },

  setIgnoreNonRaw: (v) => {
    set({ ignoreNonRaw: v });
    const { selectedFolderPath } = get();
    if (selectedFolderPath) get().selectFolder(selectedFolderPath);
  },

  addCustomPlace: (path) => {
    const name = path.split("/").pop() || path;
    const { places } = get();
    // Don't add duplicates
    if (places.some((p) => p.path === path)) {
      get().selectPlace(path);
      return;
    }
    const newPlace: Place = { name, path, type: "custom" };
    set({ places: [...places, newPlace] });
    get().selectPlace(path);
  },

  removePlace: (path) => {
    const { places, selectedPlacePath } = get();
    const filtered = places.filter((p) => p.path !== path);
    if (filtered.length === 0) return;
    set({ places: filtered });
    if (selectedPlacePath === path) {
      get().selectPlace(filtered[0].path);
    }
  },

  setLeftPanelWidth: (w) => set({ leftPanelWidth: w }),

  doImport: () => {
    const { files, importMode } = get();
    const selected = files.filter((f) => f.selected);
    const paths = selected.map((f) => f.fullpath);
    if(paths.length === 0) return;
    const isCopy = importMode === "copy";
    console.log(`[import] ${isCopy ? "copy &" : ""} importing ${paths.length} files`);
    get().closeDialog();
    const fn = isCopy ? copyAndImportImages : importImages;
    fn(paths).then((result) => {
      console.log(`[import] done: ${result.imported} imported, ${result.skipped} skipped`);
      emit("import.finished", { imported: result.imported, skipped: result.skipped });
    }).catch((err) => {
      console.error("[import] failed:", err);
    });
  },
}));

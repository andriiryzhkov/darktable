export type PlaceType = "home" | "pictures" | "mount" | "custom";

export interface Place {
  name: string;
  path: string;
  type: PlaceType;
}

export interface FolderNode {
  name: string;
  path: string;
  children: FolderNode[] | null; // null = not yet loaded (lazy)
  expanded: boolean;
}

export interface ImportFile {
  filename: string;
  fullpath: string;
  modified: number; // epoch seconds
  alreadyImported: boolean;
  selected: boolean;
}

export type SortField = "name" | "modified";
export type SortDir = "asc" | "desc";

/** Raw folder entry returned by the C listFolders binding */
export interface FolderEntry {
  name: string;
  path: string;
  hasChildren: boolean;
}

/** Raw file entry returned by the C listFiles binding */
export interface FileEntry {
  filename: string;
  fullpath: string;
  modified: number;
  alreadyImported: boolean;
}

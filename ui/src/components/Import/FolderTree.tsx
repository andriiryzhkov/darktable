import { ChevronRight } from "lucide-react";
import { useImportStore } from "../../stores/importStore";
import type { FolderNode } from "../../types/import";

function FolderTreeNode({ node, depth }: { node: FolderNode; depth: number }) {
  const selectedFolderPath = useImportStore((s) => s.selectedFolderPath);
  const selectFolder = useImportStore((s) => s.selectFolder);
  const expandFolder = useImportStore((s) => s.expandFolder);
  const collapseFolder = useImportStore((s) => s.collapseFolder);

  const hasChildren = node.children === null || (node.children && node.children.length > 0);
  const isSelected = node.path === selectedFolderPath;

  const handleChevronClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (node.expanded) {
      collapseFolder(node.path);
    } else {
      expandFolder(node.path);
    }
  };

  const handleNameClick = () => {
    selectFolder(node.path);
  };

  return (
    <>
      <div
        className="import-folder-row"
        data-selected={isSelected || undefined}
        style={{ paddingLeft: depth * 16 + 4 }}
      >
        {hasChildren ? (
          <span
            className="import-folder-chevron"
            data-expanded={node.expanded || undefined}
            onClick={handleChevronClick}
          >
            <ChevronRight size={12} />
          </span>
        ) : (
          <span className="import-folder-chevron-spacer" />
        )}
        <span className="import-folder-name" onClick={handleNameClick}>
          {node.name}
        </span>
      </div>
      {node.expanded && node.children && node.children.map((child) => (
        <FolderTreeNode key={child.path} node={child} depth={depth + 1} />
      ))}
    </>
  );
}

export default function FolderTree() {
  const folderTree = useImportStore((s) => s.folderTree);

  return (
    <div className="import-folders">
      <div className="import-section-header">
        <span className="module-section-title">folders</span>
      </div>
      <div className="import-folders-list">
        {folderTree.map((node) => (
          <FolderTreeNode key={node.path} node={node} depth={0} />
        ))}
      </div>
    </div>
  );
}

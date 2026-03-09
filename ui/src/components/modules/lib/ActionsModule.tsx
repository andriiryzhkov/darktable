import { useState, useCallback } from "react";
import LibModuleCard from "../LibModuleCard";
import ConfirmDialog from "../../ConfirmDialog";
import BauhausButton from "../../controls/BauhausButton";
import BauhausTabGroup from "../../controls/BauhausTabGroup";
import BauhausCheckbox from "../../controls/BauhausCheckbox";
import BauhausCombo from "../../controls/BauhausCombo";
import { useCatalogStore } from "../../../stores/catalogStore";
import {
  imageRemove,
  imageDelete,
  imageDuplicate,
  imageRotate,
  imageGroup,
  imageUngroup,
  imageCopyLocal,
  imageResyncLocal,
  imageRefreshExif,
  metadataPaste,
  metadataClear,
  imageSetMonochrome,
  imageMove,
  imageCopyTo,
  pickFolder,
  type MetadataFlags,
} from "../../../api/commands";
import {
  RotateCw,
  RotateCcw,
} from "lucide-react";

const TABS = ["images", "metadata"] as const;

const PASTE_MODES = ["merge", "overwrite"] as const;

export default function ActionsModule() {
  const [confirm, setConfirm] = useState<{ title: string; message: string; action: () => void } | null>(null);
  const [sourceImgId, setSourceImgId] = useState<number | null>(null);
  const [pasteMode, setPasteMode] = useState<"merge" | "overwrite">("merge");
  const [flags, setFlags] = useState<MetadataFlags>({
    ratings: true,
    colors: false,
    tags: false,
    geotags: false,
    metadata: false,
  });

  const selectedIds = useCatalogStore((s) => s.selectedIds);
  const images = useCatalogStore((s) => s.images);
  const fetchAll = useCatalogStore((s) => s.fetchAll);

  const hasSelection = selectedIds.size > 0;
  const hasOneSelected = selectedIds.size === 1;

  // Check monochrome state of selected images (DT_IMAGE_MONOCHROME_WORKFLOW = 1 << 20)
  const DT_IMAGE_MONOCHROME_WORKFLOW = 1 << 20;
  const selectedMonochrome = hasSelection
    ? images
        .filter((img) => selectedIds.has(img.id))
        .every((img) => (img.flags & DT_IMAGE_MONOCHROME_WORKFLOW) !== 0)
    : false;
  const ids = useCallback(() => Array.from(selectedIds), [selectedIds]);

  const doAction = useCallback(
    async (action: (imgids: number[]) => Promise<unknown>) => {
      await action(ids());
      fetchAll();
    },
    [ids, fetchAll],
  );

  const doFolderAction = useCallback(
    async (action: (imgids: number[], path: string) => Promise<unknown>) => {
      const path = await pickFolder();
      if (!path) return;
      await action(ids(), path);
      fetchAll();
    },
    [ids, fetchAll],
  );

  const setFlag = useCallback((key: keyof MetadataFlags, value: boolean) => {
    setFlags((f) => ({ ...f, [key]: value }));
  }, []);

  const handleCopyMetadata = useCallback(() => {
    if (hasOneSelected) {
      setSourceImgId(Array.from(selectedIds)[0]);
    }
  }, [hasOneSelected, selectedIds]);

  const handlePasteMetadata = useCallback(async () => {
    if (sourceImgId === null || !hasSelection) return;
    await metadataPaste({
      source_imgid: sourceImgId,
      imgids: ids(),
      flags,
      mode: pasteMode,
    });
    fetchAll();
  }, [sourceImgId, hasSelection, ids, flags, pasteMode, fetchAll]);

  const handleClearMetadata = useCallback(async () => {
    if (!hasSelection) return;
    await metadataClear({ imgids: ids(), flags });
    fetchAll();
  }, [hasSelection, ids, flags, fetchAll]);

  const canPaste = sourceImgId !== null && hasSelection
    && (selectedIds.size > 1 || !selectedIds.has(sourceImgId));

  return (
    <>
    <LibModuleCard title="actions on selection" description="perform various operations on the currently selected images">
      <BauhausTabGroup tabs={TABS} defaultTab="images" justify="uniform">
        {(tab) => tab === "images" ? (
          <div className="actions-grid">
            <BauhausButton label="remove" disabled={!hasSelection} onClick={() => setConfirm({
              title: selectedIds.size === 1 ? "remove image?" : "remove images?",
              message: selectedIds.size === 1
                ? "do you really want to remove 1 image from darktable\n(without deleting file on disk)?"
                : `do you really want to remove ${selectedIds.size} images from darktable\n(without deleting files on disk)?`,
              action: () => doAction(imageRemove),
            })} title="remove images from the image library, without deleting" />
            <BauhausButton label="delete (trash)" disabled={!hasSelection} onClick={() => setConfirm({
              title: selectedIds.size === 1 ? "delete image?" : "delete images?",
              message: selectedIds.size === 1
                ? "do you really want to physically delete 1 image\n(using trash if possible)?"
                : `do you really want to physically delete ${selectedIds.size} images\n(using trash if possible)?`,
              action: () => doAction(imageDelete),
            })} title="physically delete from disk (using trash if possible)" />
            <BauhausButton label="move..." disabled={!hasSelection} onClick={() => doFolderAction(imageMove)} title="move to other folder" />
            <BauhausButton label="copy..." disabled={!hasSelection} onClick={() => doFolderAction(imageCopyTo)} title="copy to other folder" />
            <BauhausButton label="create HDR" disabled={!hasSelection} title="create a high dynamic range image from selected shots" />
            <BauhausButton label="duplicate" disabled={!hasSelection} onClick={() => doAction(imageDuplicate)} title="add a duplicate to the image library, including its history stack" />
            <div className="actions-rotate-row">
              <BauhausButton icon={<RotateCcw size={12} />} disabled={!hasSelection} onClick={() => doAction((ids) => imageRotate(ids, 1))} title="rotate selected images 90 degrees CCW" />
              <BauhausButton icon={<RotateCw size={12} />} disabled={!hasSelection} onClick={() => doAction((ids) => imageRotate(ids, 0))} title="rotate selected images 90 degrees CW" />
              <BauhausButton label="reset rotation" disabled={!hasSelection} onClick={() => doAction((ids) => imageRotate(ids, 2))} title="reset rotation to EXIF data" />
            </div>
            <BauhausButton label="copy locally" disabled={!hasSelection} onClick={() => doAction(imageCopyLocal)} title="copy the image locally" />
            <BauhausButton label="resync local copy" disabled={!hasSelection} onClick={() => doAction(imageResyncLocal)} title="synchronize the image's XMP and remove the local copy" />
            <BauhausButton label="group" disabled={!hasSelection} onClick={() => doAction(imageGroup)} title="add selected images to expanded group or create a new one" />
            <BauhausButton label="ungroup" disabled={!hasSelection} onClick={() => doAction(imageUngroup)} title="remove selected images from the group" />
          </div>
        ) : (
          <div className="actions-metadata">
            <div className="actions-grid">
              <BauhausCheckbox label="ratings" checked={flags.ratings} onChange={(v) => setFlag("ratings", v)} />
              <BauhausCheckbox label="colors" checked={flags.colors} onChange={(v) => setFlag("colors", v)} />
              <BauhausCheckbox label="tags" checked={flags.tags} onChange={(v) => setFlag("tags", v)} />
              <BauhausCheckbox label="geo tags" checked={flags.geotags} onChange={(v) => setFlag("geotags", v)} />
              <BauhausCheckbox label="metadata" checked={flags.metadata} onChange={(v) => setFlag("metadata", v)} />
            </div>
            <div className="actions-metadata-buttons">
              <BauhausButton label="copy" disabled={!hasOneSelected} onClick={handleCopyMetadata} title="set the selected image as source of metadata" />
              <BauhausButton label="paste" disabled={!canPaste} onClick={handlePasteMetadata} title="paste selected metadata on selected images" />
              <BauhausButton label="clear" disabled={!hasSelection} onClick={handleClearMetadata} title="clear selected metadata on selected images" />
            </div>
            <BauhausCombo
              label="mode"
              options={[...PASTE_MODES]}
              value={pasteMode}
              onChange={(v) => setPasteMode(v as "merge" | "overwrite")}
            />
            <BauhausButton label="refresh EXIF" disabled={!hasSelection} onClick={() => doAction(imageRefreshExif)} title="update all image information to match changes to file" />
            <div className="actions-grid">
              <BauhausButton label="monochrome" disabled={!hasSelection || selectedMonochrome} onClick={() => imageSetMonochrome({ imgids: ids(), monochrome: true }).then(fetchAll)} title="set selection as monochrome images and activate monochrome workflow" />
              <BauhausButton label="color" disabled={!hasSelection || !selectedMonochrome} onClick={() => imageSetMonochrome({ imgids: ids(), monochrome: false }).then(fetchAll)} title="set selection as color images" />
            </div>
          </div>
        )}
      </BauhausTabGroup>
    </LibModuleCard>
    {confirm && (
      <ConfirmDialog
        title={confirm.title}
        message={confirm.message}
        onConfirm={() => { setConfirm(null); confirm.action(); }}
        onCancel={() => setConfirm(null)}
      />
    )}
    </>
  );
}

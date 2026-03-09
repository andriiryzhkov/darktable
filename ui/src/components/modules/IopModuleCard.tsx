import { useState, useCallback, useEffect, useRef, useMemo, Suspense, type ReactNode } from "react";
import { Copy, Crosshair, ArrowRightToLine, Workflow, ArrowRightFromLine } from "lucide-react";
import type { IopModuleDef } from "./registry";
import type { ModuleDescription } from "../../types/protocol";
import { IOP_FLAGS } from "../../types/protocol";
import BauhausButton from "../controls/BauhausButton";
import BauhausTooltip from "../controls/BauhausTooltip";
import ModuleCard from "./ModuleCard";
import PresetMenu from "./PresetMenu";
import MultiInstanceMenu from "./MultiInstanceMenu";
import { IopModuleProvider } from "./IopModuleContext";
import { useDevelopStore, getEnabledOps } from "../../stores/developStore";
import { useModuleExpanded } from "../../hooks/useModuleExpanded";

const ICON_SIZE = 10;

function ModuleDescriptionTooltip({ desc }: { desc: ModuleDescription }) {
  return (
    <div className="module-desc-tooltip">
      <div className="module-desc-main">{desc.main}</div>
      <div className="module-desc-grid">
        <Crosshair size={ICON_SIZE} strokeWidth={2.5} />
        <strong>purpose:</strong>
        <span>{desc.purpose}</span>

        <ArrowRightToLine size={ICON_SIZE} strokeWidth={2.5} />
        <strong>input:</strong>
        <span>{desc.input}</span>

        <Workflow size={ICON_SIZE} strokeWidth={2.5} />
        <strong>process:</strong>
        <span>{desc.process}</span>

        <ArrowRightFromLine size={ICON_SIZE} strokeWidth={2.5} />
        <strong>output:</strong>
        <span>{desc.output}</span>
      </div>
    </div>
  );
}

interface Props {
  module: IopModuleDef;
  /** multi_priority from the server module list (default 0 for the primary instance) */
  instance?: number;
  /** Display name suffix for multi-instance (e.g. "1", "my preset") */
  instanceName?: string;
}

export default function IopModuleCard({ module, instance: instanceProp, instanceName }: Props) {
  const instanceId = instanceProp ?? 0;
  const { open, setOpen } = useModuleExpanded("darkroom", module.op, false, "iop", instanceId);
  const historyItems = useDevelopStore((s) => s.historyItems);
  const enableModule = useDevelopStore((s) => s.enableModule);
  const resetModule = useDevelopStore((s) => s.resetModule);
  const focusModuleOp = useDevelopStore((s) => s.focusModuleOp);
  const description = useDevelopStore((s) => s.moduleDescriptions[module.op]);
  const modules = useDevelopStore((s) => s.modules);
  const newInstance = useDevelopStore((s) => s.newInstance);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const presetsRef = useRef<HTMLElement>(null);
  const multiRef = useRef<HTMLElement>(null);
  const [presetsOpen, setPresetsOpen] = useState(false);
  const [multiOpen, setMultiOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const renameRef = useRef<HTMLInputElement>(null);
  const renameInstance = useDevelopStore((s) => s.renameInstance);
  const [indicator, setIndicator] = useState<ReactNode>(null);
  const iopCtx = useMemo(() => ({ setIndicator }), []);

  // Find this module's info from the server modules list
  const moduleInfo = modules.find((m) => m.op === module.op && m.instance === instanceId);
  const supportsMulti = moduleInfo ? !(moduleInfo.flags & IOP_FLAGS.ONE_INSTANCE) : true;

  const handleStartRename = useCallback(() => {
    setRenameValue(instanceName ?? "");
    setRenaming(true);
    setMultiOpen(false);
  }, [instanceName]);

  const handleRenameSubmit = useCallback(() => {
    setRenaming(false);
    renameInstance(module.op, instanceId, renameValue.trim());
  }, [module.op, instanceId, renameValue, renameInstance]);

  useEffect(() => {
    if (renaming && renameRef.current) {
      renameRef.current.focus();
      renameRef.current.select();
    }
  }, [renaming]);

  const displayName = renaming
    ? <>{module.name} <input
        ref={renameRef}
        className="rename-instance-input"
        value={renameValue}
        onChange={(e) => setRenameValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") { e.preventDefault(); handleRenameSubmit(); }
          else if (e.key === "Escape") { e.preventDefault(); setRenaming(false); }
        }}
        onBlur={handleRenameSubmit}
        onClick={(e) => e.stopPropagation()}
      /></>
    : instanceName
      ? <>{module.name} <span className="module-instance-name">{instanceName}</span></>
      : module.name;

  useEffect(() => {
    if (focusModuleOp === module.op) {
      setOpen(true);
      useDevelopStore.setState({ focusModuleOp: null });
      requestAnimationFrame(() => {
        wrapperRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      });
    }
  }, [focusModuleOp, module.op]);

  const enabled = getEnabledOps(historyItems).has(module.op);
  const isMandatory = historyItems.some((h) => h.op === module.op && h.mandatory);

  const toggleEnabled = useCallback(() => {
    if (isMandatory) return;
    enableModule(module.op, !enabled);
  }, [module.op, enabled, enableModule, isMandatory]);

  const Component = module.component;

  return (
    <>
      <ModuleCard
        title={displayName}
        tooltip={description ? <ModuleDescriptionTooltip desc={description} /> : undefined}
        open={open}
        onToggle={setOpen}
        wrapperRef={wrapperRef}
        leftButton={{
          kind: "power",
          enabled,
          mandatory: isMandatory,
          moduleName: module.name,
          onToggle: toggleEnabled,
        }}
        indicators={indicator}
        onReset={() => resetModule(module.op)}
        resetTooltip={<span style={{ whiteSpace: "pre" }}>{"reset parameters\nctrl-click to reapply any automatic presets"}</span>}
        rightButtons={
          <span className="module-presets-wrapper" ref={multiRef as React.Ref<HTMLSpanElement>}>
            <BauhausTooltip content={<span style={{ whiteSpace: "pre" }}>{"multiple instance action\nright-click creates new instance"}</span>} placement="bottom">
              <BauhausButton
                icon={<Copy size={12} />}
                disabled={!supportsMulti}
                onClick={() => setMultiOpen((v) => !v)}
                onContextMenu={(e) => {
                  e.preventDefault();
                  if (supportsMulti) newInstance(module.op, instanceId, false);
                }}
              />
            </BauhausTooltip>
          </span>
        }
        onPresets={() => setPresetsOpen((v) => !v)}
        presetsButtonRef={presetsRef}
        presetsTooltip={<span style={{ whiteSpace: "pre" }}>{"presets\nright-click to apply on new instance"}</span>}
      >
        <IopModuleProvider value={iopCtx}>
          <Suspense fallback={null}>
            <Component />
          </Suspense>
        </IopModuleProvider>
      </ModuleCard>
      {presetsOpen && (
        <PresetMenu op={module.op} moduleName={module.name} anchorRef={presetsRef} onClose={() => setPresetsOpen(false)} />
      )}
      {multiOpen && (
        <MultiInstanceMenu op={module.op} instance={instanceId} anchorRef={multiRef} onClose={() => setMultiOpen(false)} onRename={handleStartRename} />
      )}
    </>
  );
}

import { useEffect, useRef } from "react";
import { useDevelopStore } from "../../stores/developStore";
import { MASKS_TYPE } from "../../types/protocol";
import type {
  MaskForm,
  MaskPointPath,
  MaskPointBrush,
} from "../../types/protocol";
import { inverseTransform, transformBrushPoints } from "../../lib/distortionGrid";
import {
  drawForm,
  hitTestForm,
  hitTestDragTarget,
  hitTestBrushSegment,
  isNearHandle,
} from "../../lib/maskRendering";
import type { DragTarget, DragState } from "../../lib/maskDrag";
import { getDragOps, getCreationParams } from "../../lib/maskDrag";

interface Props {
  targetRef: React.RefObject<HTMLElement | null>;
}

export default function MaskOverlay({ targetRef }: Props) {
  const distortionGrid = useDevelopStore((s) => s.distortionGrid);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hoveredIdRef = useRef<number | null>(null);
  const hoveredSegRef = useRef<number>(-1);
  const mouseRef = useRef<{ x: number; y: number } | null>(null);
  const editedPointRef = useRef<{ formid: number; index: number } | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const drawableFormsRef = useRef<MaskForm[]>([]);
  const drawRef = useRef<(() => void) | null>(null);
  // Normalized cursor position during creation — used by subscription to translate new server polylines
  const creationCursorRef = useRef<[number, number] | null>(null);

  // Recompute drawable forms when structure changes
  useEffect(() => {
    const computeDrawable = () => {
      const forms = useDevelopStore.getState().maskForms;
      const usage = useDevelopStore.getState().maskUsage;
      const selId = useDevelopStore.getState().selectedMaskId;
      const creating = useDevelopStore.getState().creatingMaskId;
      const show = useDevelopStore.getState().showMasks;

      const result: MaskForm[] = [];
      if (show) {
        const usedFormIds = new Set<number>();
        for (const u of usage) {
          const group = forms.find((f) => f.formid === u.mask_id);
          if (group?.children) {
            for (const c of group.children) usedFormIds.add(c.formid);
          }
        }
        for (const form of forms) {
          if (usedFormIds.has(form.formid)) result.push(form);
        }
      }
      const selForm = selId !== null ? forms.find((f) => f.formid === selId) ?? null : null;
      if (selForm && !result.some((f) => f.formid === selForm.formid)) {
        result.push(selForm);
      }
      if (creating !== null && !result.some((f) => f.formid === creating)) {
        const creatingForm = forms.find((f) => f.formid === creating);
        if (creatingForm) result.push(creatingForm);
      }
      drawableFormsRef.current = result;
    };
    computeDrawable();
    // Subscribe to store changes for lightweight redraw (no event listener teardown)
    const unsub = useDevelopStore.subscribe((state, prev) => {
      if (state.maskForms !== prev.maskForms || state.selectedMaskId !== prev.selectedMaskId
          || state.creatingMaskId !== prev.creatingMaskId || state.showMasks !== prev.showMasks) {
        computeDrawable();
        // During creation, when server returns new polylines (e.g. from slider preview),
        // update position to match cursor so next draw is correct
        if (state.creatingMaskId !== null && state.maskForms !== prev.maskForms) {
          const cursor = creationCursorRef.current;
          const outPos: [number, number] = cursor ?? [0.5, 0.5];
          const grid = useDevelopStore.getState().distortionGrid;
          if (grid) {
            const form = drawableFormsRef.current.find((f) => f.formid === state.creatingMaskId);
            if (form) {
              const rawPos = inverseTransform(grid, outPos[0], outPos[1]);
              const ops = getDragOps(form);
              if (ops) {
                ops.setPosition(form, [rawPos[0], rawPos[1]]);
              }
            }
          }
        }
        drawRef.current?.();
      }
    });
    return unsub;
  }, []); // stable — reads from store directly

  useEffect(() => {
    const target = targetRef.current;
    const canvas = canvasRef.current;
    if (!target || !canvas) return;

    const draw = () => {
      // Check if there's anything to draw (read latest state)
      const { maskForms, selectedMaskId: selId, showMasks: show,
              creationTool: cTool, creatingMaskId: cMaskId } = useDevelopStore.getState();
      const selForm = selId !== null ? maskForms.find((f) => f.formid === selId) ?? null : null;
      const hasAnythingToDraw = (show && drawableFormsRef.current.length > 0) || selForm || cTool || cMaskId;
      if (!hasAnythingToDraw) {
        canvas.style.display = "none";
        canvas.style.pointerEvents = "none";
        return;
      }
      canvas.style.pointerEvents = "auto";
      const tr = target.getBoundingClientRect();
      const container = target.closest(".preview-container");
      if (!container) return;
      const pr = container.getBoundingClientRect();

      // Compute the actual rendered content area for object-contain
      const imgCanvas = target as HTMLCanvasElement;
      const iw = imgCanvas.width;
      const ih = imgCanvas.height;
      const cssW = tr.width;
      const cssH = tr.height;

      let contentW: number, contentH: number, contentX: number, contentY: number;
      if (iw > 0 && ih > 0) {
        const scale = Math.min(cssW / iw, cssH / ih);
        contentW = iw * scale;
        contentH = ih * scale;
        contentX = (tr.left - pr.left) + (cssW - contentW) / 2;
        contentY = (tr.top - pr.top) + (cssH - contentH) / 2;
      } else {
        contentW = cssW;
        contentH = cssH;
        contentX = tr.left - pr.left;
        contentY = tr.top - pr.top;
      }

      const w = Math.round(contentW);
      const h = Math.round(contentH);
      if (w === 0 || h === 0) return;

      canvas.style.display = "block";
      canvas.style.left = `${contentX}px`;
      canvas.style.top = `${contentY}px`;
      canvas.width = w;
      canvas.height = h;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, w, h);

      const m = mouseRef.current;
      const ep = editedPointRef.current;
      for (const form of drawableFormsRef.current) {
        const hovered = hoveredIdRef.current === form.formid;
        const seg = hovered ? hoveredSegRef.current : -1;
        drawForm(ctx, w, h, form, hovered, m?.x ?? null, m?.y ?? null, ep, seg, distortionGrid);
      }

      // Compute cursor from current state — single source of truth
      if (dragRef.current) {
        const dk = dragRef.current.target.kind;
        canvas.style.cursor = dk === "center" ? "grabbing" : dk === "rotate" ? "alias" : "crosshair";
      } else if (cTool || cMaskId) {
        canvas.style.cursor = "crosshair";
      } else if (m && selId !== null) {
        const selF = drawableFormsRef.current.find((f) => f.formid === selId);
        if (selF) {
          const hit = hitTestDragTarget(ctx, w, h, selF, m.x, m.y, distortionGrid);
          canvas.style.cursor = hit ? (hit.kind === "center" ? "grab" : hit.kind === "rotate" ? "alias" : "crosshair") : "";
        } else {
          canvas.style.cursor = "";
        }
      } else {
        canvas.style.cursor = "";
      }
    };

    drawRef.current = draw;

    const { createMask, updateMask, resetCreation, requestPreview,
            saveCreation, cancelCreation, previewMaskParam } = useDevelopStore.getState();

    const onMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      mouseRef.current = { x: px, y: py };
      const w = canvas.width;
      const h = canvas.height;

      // Creation mode: update form position to follow cursor
      const currentCreatingId = useDevelopStore.getState().creatingMaskId;
      if (currentCreatingId !== null) {
        const cx = px / w;
        const cy = py / h;
        creationCursorRef.current = [cx, cy];
        const grid = useDevelopStore.getState().distortionGrid;
        if (grid) {
          const form = drawableFormsRef.current.find((f) => f.formid === currentCreatingId);
          if (form) {
            const rawPos = inverseTransform(grid, cx, cy);
            const ops = getDragOps(form);
            if (ops) {
              ops.setPosition(form, [rawPos[0], rawPos[1]]);
              // Background server sync — send raw-space position
              const posParam = ops.previewParams(form, "center");
              previewMaskParam(currentCreatingId, posParam);
            }
          }
        }
        draw();
        return;
      }

      // Handle drag in progress
      const drag = dragRef.current;
      if (drag) {
        const grid = useDevelopStore.getState().distortionGrid;
        if (!grid) { draw(); return; }
        const form = drawableFormsRef.current.find((f) => f.formid === drag.target.formid);
        if (!form) return;
        const ops = getDragOps(form);
        if (!ops) { draw(); return; }
        const rawMouse = inverseTransform(grid, px / w, py / h);
        ops.updateDrag(form, drag, [rawMouse[0], rawMouse[1]], px, py, w, h, grid);
        // Live server sync — only send changed params
        const previewP = ops.previewParams(form, drag.target.kind);
        previewMaskParam(drag.target.formid, previewP);
        draw();
        return;
      }

      // Normal hover detection
      let newHovered: number | null = null;
      let newSeg = -1;
      for (const form of drawableFormsRef.current) {
        const ctx2 = canvas.getContext("2d");
        if (ctx2 && hitTestForm(ctx2, w, h, form, px, py, distortionGrid)) {
          newHovered = form.formid;
          const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
          if (baseType === MASKS_TYPE.BRUSH && distortionGrid && form.points) {
            const tPts = transformBrushPoints(distortionGrid, form.points as MaskPointBrush[]);
            newSeg = hitTestBrushSegment(ctx2, w, h, tPts, px, py);
          }
          break;
        }
      }

      hoveredIdRef.current = newHovered;
      hoveredSegRef.current = newSeg;

      draw();
    };

    const onMouseLeave = () => {
      mouseRef.current = null;
      hoveredIdRef.current = null;
      hoveredSegRef.current = -1;
      // During creation, center the mask when mouse leaves the image area
      const currentCreatingId = useDevelopStore.getState().creatingMaskId;
      if (currentCreatingId !== null) {
        creationCursorRef.current = null;
        const grid = useDevelopStore.getState().distortionGrid;
        if (grid) {
          const form = drawableFormsRef.current.find((f) => f.formid === currentCreatingId);
          if (form) {
            const rawCenter = inverseTransform(grid, 0.5, 0.5);
            const ops = getDragOps(form);
            if (ops) {
              ops.setPosition(form, [rawCenter[0], rawCenter[1]]);
              const posParam = ops.previewParams(form, "center");
              previewMaskParam(currentCreatingId, posParam);
            }
          }
        }
      }
      draw();
    };

    const onMouseDown = (e: MouseEvent) => {
      if (e.button !== 0) return;
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const w = canvas.width;
      const h = canvas.height;

      // Creation mode: click to save mask position
      const currentCreatingId = useDevelopStore.getState().creatingMaskId;
      if (currentCreatingId !== null) {
        const cx = px / w;
        const cy = py / h;
        creationCursorRef.current = [cx, cy];
        const grid = useDevelopStore.getState().distortionGrid;
        const rawPos = grid ? inverseTransform(grid, cx, cy) as [number, number] : [cx, cy] as [number, number];
        saveCreation(rawPos);
        return;
      }

      // Creation mode (from blending toolbar): click to create mask
      const tool = useDevelopStore.getState().creationTool;
      const mod = useDevelopStore.getState().creationModule;
      if (tool && mod) {
        const outCx = px / w;
        const outCy = py / h;
        const grid = useDevelopStore.getState().distortionGrid;
        const position = grid
          ? inverseTransform(grid, outCx, outCy) as [number, number]
          : [outCx, outCy] as [number, number];
        const params: Record<string, unknown> = {
          op: mod.op,
          instance: mod.instance,
          ...getCreationParams(tool, position),
        };
        resetCreation();
        createMask(tool, params).then((formid) => {
          if (formid) requestPreview();
        });
        return;
      }

      // Try to start a drag on a form — returns true if drag started
      const tryStartDrag = (form: MaskForm, ctx2: CanvasRenderingContext2D): boolean => {
        const hitTarget = hitTestDragTarget(ctx2, w, h, form, px, py, distortionGrid);
        if (!hitTarget) return false;
        const ops = getDragOps(form);
        if (!ops) return false;
        dragRef.current = ops.initDrag(form, hitTarget, px, py, w, h, distortionGrid!);
        return true;
      };

      // Editing mode: check for drag targets on the selected mask
      const ctx2 = canvas.getContext("2d");
      const currentSelectedId = useDevelopStore.getState().selectedMaskId;
      if (ctx2 && currentSelectedId !== null) {
        const selForm = drawableFormsRef.current.find((f) => f.formid === currentSelectedId);
        if (selForm) {
          const baseType = selForm.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
          // Ctrl+click on ellipse or gradient → rotation mode
          if (e.ctrlKey && (baseType === MASKS_TYPE.ELLIPSE || baseType === MASKS_TYPE.GRADIENT) && hitTestForm(ctx2, w, h, selForm, px, py, distortionGrid)) {
            const grid = useDevelopStore.getState().distortionGrid;
            if (grid) {
              const ops = getDragOps(selForm);
              if (ops) {
                const rotateTarget: DragTarget = { kind: "rotate", formid: selForm.formid };
                dragRef.current = ops.initDrag(selForm, rotateTarget, px, py, w, h, grid);
                e.preventDefault();
                return;
              }
            }
          }
          if (tryStartDrag(selForm, ctx2)) {
            e.preventDefault();
            return;
          }
        }
      }

      // Click on a form to select it — also start drag if inside border
      if (ctx2) {
        for (const form of drawableFormsRef.current) {
          if (hitTestForm(ctx2, w, h, form, px, py, distortionGrid)) {
            useDevelopStore.getState().selectMask(form.formid);
            if (tryStartDrag(form, ctx2)) {
              e.preventDefault();
              return;
            }
            draw();
            return;
          }
        }
        // Clicked on empty space — deselect
        if (currentSelectedId !== null) {
          useDevelopStore.getState().selectMask(null);
          draw();
        }
      }
    };

    const onMouseUp = (e: MouseEvent) => {
      const drag = dragRef.current;
      if (!drag) {
        // Path corner handle toggle on click (only if not drag)
        const rect = canvas.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const py = e.clientY - rect.top;
        const w = canvas.width;
        const h = canvas.height;
        for (const form of drawableFormsRef.current) {
          const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
          if (baseType !== MASKS_TYPE.PATH || !form.points) continue;
          const pts = form.points as MaskPointPath[];
          for (let i = 0; i < pts.length; i++) {
            if (isNearHandle(pts[i].corner[0] * w, pts[i].corner[1] * h, px, py)) {
              const cur = editedPointRef.current;
              if (cur && cur.formid === form.formid && cur.index === i) {
                editedPointRef.current = null;
              } else {
                editedPointRef.current = { formid: form.formid, index: i };
              }
              draw();
              return;
            }
          }
        }
        if (editedPointRef.current) {
          editedPointRef.current = null;
          draw();
        }
        return;
      }

      // Commit the drag to the server
      dragRef.current = null;
      const form = drawableFormsRef.current.find((f) => f.formid === drag.target.formid);
      if (!form) return;
      const ops = getDragOps(form);
      if (!ops) return;
      updateMask(form.formid, ops.commitParams(form)).then(() => requestPreview());
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        const creating = useDevelopStore.getState().creatingMaskId;
        if (creating) {
          creationCursorRef.current = null;
          cancelCreation();
          return;
        }
        const tool = useDevelopStore.getState().creationTool;
        if (tool) {
          resetCreation();
        }
      }
    };

    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mouseleave", onMouseLeave);
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mouseup", onMouseUp);
    window.addEventListener("keydown", onKeyDown);

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(target);
    const mo2 = new MutationObserver(draw);
    mo2.observe(target, { attributes: true, attributeFilter: ["style"] });
    return () => {
      drawRef.current = null;
      ro.disconnect();
      mo2.disconnect();
      canvas.removeEventListener("mousemove", onMouseMove);
      canvas.removeEventListener("mouseleave", onMouseLeave);
      canvas.removeEventListener("mousedown", onMouseDown);
      canvas.removeEventListener("mouseup", onMouseUp);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [targetRef, distortionGrid]);

  return <canvas ref={canvasRef} className="mask-overlay" />;
}

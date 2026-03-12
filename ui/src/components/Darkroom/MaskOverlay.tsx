import { useEffect, useRef } from "react";
import { useDevelopStore } from "../../stores/developStore";
import { MASKS_TYPE } from "../../types/protocol";
import type {
  MaskForm,
  MaskPointPath,
  MaskPointBrush,
} from "../../types/protocol";
import { inverseTransform, forwardTransform, transformBrushPoints, transformPathPoints } from "../../lib/distortionGrid";
import {
  drawForm,
  drawHandle,
  dualStroke,
  dualStrokePath2D,
  hitTestForm,
  hitTestDragTarget,
  hitTestPathCtrlHandle,
  hitTestBrushSegment,
  isNearHandle,
  findNearestPathSegment,
  splitBezierAt,
  pathWindingCW,
  buildBorderPolyline,
  computeBorderAnchors,
} from "../../lib/maskRendering";
import type { DragTarget, DragState } from "../../lib/maskDrag";
import { getDragOps, getCreationParams, initPathCtrlPoints } from "../../lib/maskDrag";

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
  // Accumulated points during multi-click path creation
  const pathCreationRef = useRef<MaskPointPath[] | null>(null);

  // Recompute drawable forms when structure changes
  useEffect(() => {
    const computeDrawable = () => {
      const forms = useDevelopStore.getState().maskForms;
      const selId = useDevelopStore.getState().selectedMaskId;
      const creating = useDevelopStore.getState().creatingMaskId;
      const show = useDevelopStore.getState().showMasks;

      const result: MaskForm[] = [];
      // Only draw the selected mask form (not all group children)
      if (show && selId !== null) {
        const selForm = forms.find((f) => f.formid === selId);
        if (selForm) result.push(selForm);
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
      // Clear path creation state when creation tool is deactivated
      if (state.creationTool !== prev.creationTool && state.creationTool === null) {
        pathCreationRef.current = null;
        drawRef.current?.();
      }
      if (state.maskForms !== prev.maskForms || state.selectedMaskId !== prev.selectedMaskId
          || state.creatingMaskId !== prev.creatingMaskId || state.showMasks !== prev.showMasks) {
        // Clear path control-handle editing when switching to a different mask
        if (state.selectedMaskId !== prev.selectedMaskId) {
          editedPointRef.current = null;
        }
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

      // Draw path creation preview — use regular colors, dynamic cursor feedback, and feather
      const pcPts = pathCreationRef.current;
      if (pcPts && pcPts.length > 0 && cTool === "path") {
        const grid = useDevelopStore.getState().distortionGrid;

        // Build a virtual closed path: existing points + cursor as virtual next point
        // This gives dynamic feedback showing what the path will look like
        const virtualPts: MaskPointPath[] = pcPts.map((p) => ({
          corner: [p.corner[0], p.corner[1]] as [number, number],
          ctrl1: [p.ctrl1[0], p.ctrl1[1]] as [number, number],
          ctrl2: [p.ctrl2[0], p.ctrl2[1]] as [number, number],
          border: [p.border[0], p.border[1]] as [number, number],
          state: p.state,
        }));

        // Add cursor position as virtual point for dynamic preview
        if (m && grid) {
          const rawCursor = inverseTransform(grid, m.x / w, m.y / h);
          virtualPts.push({
            corner: [rawCursor[0], rawCursor[1]] as [number, number],
            ctrl1: [-1, -1] as [number, number],
            ctrl2: [-1, -1] as [number, number],
            border: [pcPts[pcPts.length - 1].border[0], pcPts[pcPts.length - 1].border[1]] as [number, number],
            state: 1, // NORMAL — auto catmull-rom
          });
        }

        // Recompute catmull-rom ctrl points for the virtual path
        if (virtualPts.length >= 2) {
          // Reset ctrl points to -1 for NORMAL state points so initPathCtrlPoints recomputes them
          for (const vp of virtualPts) {
            if (vp.state & 1) {
              vp.ctrl1 = [-1, -1];
              vp.ctrl2 = [-1, -1];
            }
          }
          initPathCtrlPoints(virtualPts);
        }

        // Transform to output space
        const tPts: MaskPointPath[] = virtualPts.map((p) => {
          const tc = grid ? forwardTransform(grid, p.corner[0], p.corner[1]) : [p.corner[0], p.corner[1]] as [number, number];
          const c1: [number, number] = p.ctrl1[0] !== -1 && grid
            ? forwardTransform(grid, p.ctrl1[0], p.ctrl1[1])
            : p.ctrl1[0] !== -1 ? [p.ctrl1[0], p.ctrl1[1]] : tc;
          const c2: [number, number] = p.ctrl2[0] !== -1 && grid
            ? forwardTransform(grid, p.ctrl2[0], p.ctrl2[1])
            : p.ctrl2[0] !== -1 ? [p.ctrl2[0], p.ctrl2[1]] : tc;
          return { corner: tc, ctrl1: c1, ctrl2: c2, border: p.border, state: p.state };
        });

        if (tPts.length >= 3) {
          // Draw main path — closed bezier with dual stroke
          ctx.beginPath();
          ctx.moveTo(tPts[0].corner[0] * w, tPts[0].corner[1] * h);
          for (let i = 0; i < tPts.length; i++) {
            const curr = tPts[i];
            const next = tPts[(i + 1) % tPts.length];
            ctx.bezierCurveTo(
              curr.ctrl2[0] * w, curr.ctrl2[1] * h,
              next.ctrl1[0] * w, next.ctrl1[1] * h,
              next.corner[0] * w, next.corner[1] * h,
            );
          }
          ctx.closePath();
          dualStroke(ctx, false, true);

          // Draw feather border
          const cw = pathWindingCW(tPts);
          const borderPath = buildBorderPolyline(tPts, w, h, cw);
          dualStrokePath2D(ctx, borderPath, true, true);

          // Draw border anchor handles
          const borderAnchors = computeBorderAnchors(tPts, w, h, cw);
          for (let i = 0; i < pcPts.length; i++) {
            drawHandle(ctx, borderAnchors[i].x, borderAnchors[i].y, null, null);
          }
        } else if (tPts.length === 2) {
          // Just 2 points — draw line segment
          ctx.beginPath();
          ctx.moveTo(tPts[0].corner[0] * w, tPts[0].corner[1] * h);
          ctx.bezierCurveTo(
            tPts[0].ctrl2[0] * w, tPts[0].ctrl2[1] * h,
            tPts[1].ctrl1[0] * w, tPts[1].ctrl1[1] * h,
            tPts[1].corner[0] * w, tPts[1].corner[1] * h,
          );
          dualStroke(ctx, false, true);
        }

        // Draw corner handles for placed points (not the virtual cursor point)
        for (let i = 0; i < pcPts.length; i++) {
          drawHandle(ctx, tPts[i].corner[0] * w, tPts[i].corner[1] * h, null, null);
        }
      }

      // Compute cursor from current state — single source of truth
      if (dragRef.current) {
        const dk = dragRef.current.target.kind;
        canvas.style.cursor = dk === "center" ? "grabbing"
          : dk === "pathCorner" ? "move"
          : dk === "rotate" ? "alias" : "crosshair";
      } else if (cTool || cMaskId) {
        canvas.style.cursor = "crosshair";
      } else if (m && selId !== null) {
        const selF = drawableFormsRef.current.find((f) => f.formid === selId);
        if (selF) {
          const hit = hitTestDragTarget(ctx, w, h, selF, m.x, m.y, distortionGrid);
          canvas.style.cursor = hit
            ? (hit.kind === "center" ? "grab"
              : hit.kind === "pathCorner" ? "move"
              : hit.kind === "rotate" ? "alias" : "crosshair")
            : "";
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
        drag.shiftKey = e.shiftKey;
        ops.updateDrag(form, drag, [rawMouse[0], rawMouse[1]], px, py, w, h, grid);
        drag.moved = true;
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
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const w = canvas.width;
      const h = canvas.height;

      // Path creation: right-click to finish
      if (e.button === 2) {
        const pts = pathCreationRef.current;
        if (pts) {
          e.preventDefault();
          if (pts.length >= 3) {
            // Reset ctrl to -1 then recompute via catmull-rom
            for (const p of pts) {
              if (p.state & 1) { p.ctrl1 = [-1, -1]; p.ctrl2 = [-1, -1]; }
            }
            initPathCtrlPoints(pts);
            const mod = useDevelopStore.getState().creationModule;
            const params: Record<string, unknown> = {
              points: pts.map((p) => ({ ...p, corner: [...p.corner], ctrl1: [...p.ctrl1], ctrl2: [...p.ctrl2], border: [...p.border] })),
            };
            if (mod) { params.op = mod.op; params.instance = mod.instance; }
            pathCreationRef.current = null;
            resetCreation();
            createMask("path", params).then((formid) => {
              if (formid) requestPreview();
            });
          } else {
            pathCreationRef.current = null;
            resetCreation();
          }
          draw();
          return;
        }
        return;
      }

      if (e.button !== 0) return;

      // Path creation: left-click to add point
      const tool = useDevelopStore.getState().creationTool;
      if (tool === "path") {
        const grid = useDevelopStore.getState().distortionGrid;
        const rawPos = grid
          ? inverseTransform(grid, px / w, py / h)
          : [px / w, py / h] as [number, number];
        if (!pathCreationRef.current) pathCreationRef.current = [];
        const pts = pathCreationRef.current;
        // Ctrl+click: make the PREVIOUS point sharp
        if (e.ctrlKey && pts.length > 0) {
          const prev = pts[pts.length - 1];
          prev.ctrl1 = [prev.corner[0], prev.corner[1]];
          prev.ctrl2 = [prev.corner[0], prev.corner[1]];
          prev.state = 2; // USER (sharp)
        }
        // Add new point
        pts.push({
          corner: [rawPos[0], rawPos[1]],
          ctrl1: [-1, -1],
          ctrl2: [-1, -1],
          border: [0.05, 0.05],
          state: 1, // NORMAL
        });
        // Recompute control points for smooth curves
        if (pts.length >= 2) {
          for (const p of pts) {
            if (p.state & 1) { p.ctrl1 = [-1, -1]; p.ctrl2 = [-1, -1]; }
          }
          initPathCtrlPoints(pts);
        }
        draw();
        return;
      }

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

          // Path: check control point handles when a point is being edited
          if (baseType === MASKS_TYPE.PATH && editedPointRef.current && distortionGrid) {
            const ep = editedPointRef.current;
            if (ep.formid === selForm.formid) {
              const ctrlHit = hitTestPathCtrlHandle(w, h, selForm, ep.index, px, py, distortionGrid);
              if (ctrlHit) {
                const ops = getDragOps(selForm);
                if (ops) {
                  dragRef.current = ops.initDrag(selForm, ctrlHit, px, py, w, h, distortionGrid);
                  e.preventDefault();
                  return;
                }
              }
            }
          }

          // Ctrl+click on path edge → insert new control point
          if (e.ctrlKey && baseType === MASKS_TYPE.PATH && distortionGrid && selForm.points) {
            const pts = selForm.points as MaskPointPath[];
            const tPts = transformPathPoints(distortionGrid, pts);
            const hit = findNearestPathSegment(tPts, w, h, px, py);
            if (hit) {
              const { segIdx, t } = hit;
              const curr = pts[segIdx];
              const next = pts[(segIdx + 1) % pts.length];
              // Split the bezier segment at t using de Casteljau
              const { left, right } = splitBezierAt(
                curr.corner, curr.ctrl2, next.ctrl1, next.corner, t,
              );
              // Interpolate border
              const borderVal = curr.border[1] + (next.border[0] - curr.border[1]) * t;
              // Create new point from the split
              const newPoint: MaskPointPath = {
                corner: left[3],
                ctrl1: left[2],
                ctrl2: right[1],
                border: [borderVal, borderVal],
                state: 0,
              };
              // Update existing points' control handles from the split
              curr.ctrl2 = left[1];
              next.ctrl1 = right[2];
              // Insert new point after segIdx
              pts.splice(segIdx + 1, 0, newPoint);
              // Commit to server
              updateMask(selForm.formid, { points: pts.map((p) => ({ ...p, corner: [...p.corner], ctrl1: [...p.ctrl1], ctrl2: [...p.ctrl2], border: [...p.border] })) })
                .then(() => requestPreview());
              editedPointRef.current = { formid: selForm.formid, index: segIdx + 1 };
              e.preventDefault();
              draw();
              return;
            }
          }

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
        const grid = useDevelopStore.getState().distortionGrid;
        for (const form of drawableFormsRef.current) {
          const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
          if (baseType !== MASKS_TYPE.PATH || !form.points || !grid) continue;
          const pts = form.points as MaskPointPath[];
          const tPts = transformPathPoints(grid, pts);
          for (let i = 0; i < tPts.length; i++) {
            if (isNearHandle(tPts[i].corner[0] * w, tPts[i].corner[1] * h, px, py)) {
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

      // Pure click on a path corner (no movement) → toggle ctrl handle editing
      dragRef.current = null;
      if (!drag.moved && drag.target.kind === "pathCorner") {
        const idx = drag.target.pointIndex;
        const cur = editedPointRef.current;
        if (cur && cur.formid === drag.target.formid && cur.index === idx) {
          editedPointRef.current = null;
        } else {
          editedPointRef.current = { formid: drag.target.formid, index: idx };
        }
        draw();
        return;
      }

      // Commit the drag to the server
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
          pathCreationRef.current = null;
          resetCreation();
        }
      }

      // Delete/Backspace: remove the currently edited path point
      if ((e.key === "Delete" || e.key === "Backspace") && editedPointRef.current) {
        const ep = editedPointRef.current;
        const form = drawableFormsRef.current.find((f) => f.formid === ep.formid);
        if (form) {
          const baseType = form.type & ~(MASKS_TYPE.CLONE | MASKS_TYPE.NON_CLONE);
          if (baseType === MASKS_TYPE.PATH && form.points) {
            const pts = form.points as MaskPointPath[];
            // Need at least 3 points to keep a valid path after removal
            if (pts.length > 3) {
              pts.splice(ep.index, 1);
              editedPointRef.current = null;
              updateMask(form.formid, { points: pts.map((p) => ({ ...p, corner: [...p.corner], ctrl1: [...p.ctrl1], ctrl2: [...p.ctrl2], border: [...p.border] })) })
                .then(() => requestPreview());
              draw();
              e.preventDefault();
            }
          }
        }
      }
    };

    const onContextMenu = (e: MouseEvent) => {
      // Prevent browser context menu during path creation
      if (pathCreationRef.current || useDevelopStore.getState().creationTool === "path") {
        e.preventDefault();
      }
    };

    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mouseleave", onMouseLeave);
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("contextmenu", onContextMenu);
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
      canvas.removeEventListener("contextmenu", onContextMenu);
      canvas.removeEventListener("mouseup", onMouseUp);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [targetRef, distortionGrid]);

  return <canvas ref={canvasRef} className="mask-overlay" />;
}

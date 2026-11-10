import { iconNodes } from "./icon-nodes.mjs";
const sprite = new URL("./icons.svg", import.meta.url).href;
export function icon(name, className = "") {
  if (!iconNodes[name]) throw new Error(`Unknown Lucide icon: ${name}`);
  return `<svg class="icon ${className}" aria-hidden="true" focusable="false" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><use href="${sprite}#${name}"></use></svg>`;
}
export function buttonLabel(button, name, label) {
  button.innerHTML = icon(name) + `<span>${label}</span>`;
}
export function canvasIcon(ctx, name, x, y, size, color) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(size / 24, size / 24);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  for (const [tag, attrs] of iconNodes[name])
    if (tag === "path") ctx.stroke(new Path2D(attrs.d));
  ctx.restore();
}

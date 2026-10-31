// Pictures the user pastes or drops on the composer (a screenshot, a figure saved from the web) become image chips.
// Each is redrawn as a PNG no longer than 1568 px on its long edge: what a Claude-class model sees anyway
// (economy.ts imageTokens), so a 5K screenshot does not ride along as 10 MB of base64.
import type { ContextChip } from "../types.ts";
import { env } from "./dom.ts";

const MAX_EDGE = 1568;
const MAX_IMAGES = 5;
let seq = 0;

/** The image files in a paste or drop, at most five. */
export const imageFiles = (data: DataTransfer | null): File[] =>
  [...(data?.files ?? [])].filter((f) => f.type.startsWith("image/")).slice(0, MAX_IMAGES);

async function toPng(file: File): Promise<string> {
  const bmp = await env.win.createImageBitmap(file);
  const k = Math.min(1, MAX_EDGE / Math.max(bmp.width, bmp.height));
  const canvas = env.doc.createElementNS("http://www.w3.org/1999/xhtml", "canvas") as HTMLCanvasElement;
  canvas.width = Math.max(1, Math.round(bmp.width * k));
  canvas.height = Math.max(1, Math.round(bmp.height * k));
  canvas.getContext("2d")!.drawImage(bmp, 0, 0, canvas.width, canvas.height);
  bmp.close();
  return canvas.toDataURL("image/png").split(",")[1] ?? "";
}

/** One chip per readable picture; a file that does not decode is skipped. */
export async function imageChips(files: File[]): Promise<ContextChip[]> {
  const out: ContextChip[] = [];
  for (const f of files) {
    try {
      const data = await toPng(f);
      // Pasted screenshots are all called "image.png": only a real file name is worth showing.
      const label = f.name && !/^image\.\w+$/i.test(f.name) ? f.name : "Image";
      if (data) out.push({ id: `image:${++seq}`, kind: "image", label, auto: false, pinned: false, ref: { libraryID: 0 }, image: { mime: "image/png", data } });
    } catch { /* not an image the browser can read */ }
  }
  return out;
}

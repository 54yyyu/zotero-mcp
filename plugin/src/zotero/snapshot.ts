// Window snapshots for the harness: what the panel really looks like in Gecko, no screen-recording permission needed.
const XHTML = "http://www.w3.org/1999/xhtml";

export async function snapshot(win: any, path: string): Promise<string> {
  const bitmap = await win.browsingContext.currentWindowGlobal.drawSnapshot(null, 1, "rgb(255, 255, 255)");
  const canvas = win.document.createElementNS(XHTML, "canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
  const blob: Blob = await new Promise((r) => canvas.toBlob(r, "image/png"));
  await IOUtils.write(path, new Uint8Array(await blob.arrayBuffer()));
  return path;
}

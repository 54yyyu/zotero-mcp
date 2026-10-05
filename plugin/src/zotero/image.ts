// The background picture. The user picks one; it is drawn at most 1600 px on its long edge into a JPEG (a few hundred KB
// at most), kept as <dataDir>/background.jpg (the prefs hold only its name), and handed to the panel as a data: URL,
// which is the one kind of image URL Zotero loads into the panel's stylesheet.
const MAX_EDGE = 1600;

export function createImages(win: any, dataDir: string) {
  const file = PathUtils.join(dataDir, "background.jpg");
  const dataUrl = (bytes: Uint8Array): string => {
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return `data:image/jpeg;base64,${win.btoa(s)}`;
  };

  /** The part after the picker: read, downscale, save, return. */
  async function importImage(path: string): Promise<{ name: string; dataUrl: string }> {
    let bmp: any;
    try { bmp = await win.createImageBitmap(new win.Blob([await IOUtils.read(path)])); } catch {
      throw new Error("Zotero can't open that file as an image. Choose a PNG, JPEG, GIF, WebP or AVIF.");
    }
    const k = Math.min(1, MAX_EDGE / Math.max(bmp.width, bmp.height));
    const c = win.document.createElementNS("http://www.w3.org/1999/xhtml", "canvas");
    c.width = Math.max(1, Math.round(bmp.width * k));
    c.height = Math.max(1, Math.round(bmp.height * k));
    const g = c.getContext("2d");
    g.fillStyle = "#fff"; // a transparent PNG gets a white ground, not JPEG's black
    g.fillRect(0, 0, c.width, c.height);
    g.drawImage(bmp, 0, 0, c.width, c.height);
    bmp.close();
    const blob: Blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.85));
    const bytes = new Uint8Array(await blob.arrayBuffer());
    await IOUtils.makeDirectory(dataDir, { createAncestors: true, ignoreExisting: true });
    await IOUtils.write(file, bytes);
    return { name: PathUtils.filename(path), dataUrl: dataUrl(bytes) };
  }

  return {
    importImage,
    async choose(): Promise<{ name: string; dataUrl: string } | null> {
      const fp = Cc["@mozilla.org/filepicker;1"].createInstance(Ci.nsIFilePicker);
      fp.init(win.browsingContext, "Choose a background image", Ci.nsIFilePicker.modeOpen);
      fp.appendFilters(Ci.nsIFilePicker.filterImages);
      const result: number = await new Promise((r) => fp.open(r));
      return result === Ci.nsIFilePicker.returnOK ? importImage(fp.file.path) : null;
    },
    async load(): Promise<string | null> {
      try { return dataUrl(await IOUtils.read(file)); } catch { return null; }
    },
    remove: (): Promise<void> => IOUtils.remove(file, { ignoreAbsent: true }),
  };
}

import sharp from "sharp";
import { readFileSync, writeFileSync } from "node:fs";

const svgPath = new URL("../public/favicon.svg", import.meta.url);
const icoPath = new URL("../public/favicon.ico", import.meta.url);
const svg = readFileSync(svgPath);

const sizes = [16, 32, 48];

const pngs = await Promise.all(
  sizes.map((size) => sharp(svg, { density: 384 }).resize(size, size).png().toBuffer())
);

// Minimal ICO container embedding PNG data directly (the modern, widely
// supported "PNG-in-ICO" format), so no BMP encoding is needed.
const headerSize = 6;
const dirEntrySize = 16;
let offset = headerSize + dirEntrySize * pngs.length;

const header = Buffer.alloc(headerSize);
header.writeUInt16LE(0, 0); // reserved
header.writeUInt16LE(1, 2); // type: icon
header.writeUInt16LE(pngs.length, 4); // image count

const dirEntries = [];
for (let i = 0; i < pngs.length; i++) {
  const size = sizes[i];
  const png = pngs[i];
  const entry = Buffer.alloc(dirEntrySize);
  entry.writeUInt8(size === 256 ? 0 : size, 0); // width (0 means 256)
  entry.writeUInt8(size === 256 ? 0 : size, 1); // height
  entry.writeUInt8(0, 2); // color palette
  entry.writeUInt8(0, 3); // reserved
  entry.writeUInt16LE(1, 4); // color planes
  entry.writeUInt16LE(32, 6); // bits per pixel
  entry.writeUInt32LE(png.length, 8); // image data size
  entry.writeUInt32LE(offset, 12); // image data offset
  dirEntries.push(entry);
  offset += png.length;
}

const ico = Buffer.concat([header, ...dirEntries, ...pngs]);
writeFileSync(icoPath, ico);
console.log(`Wrote ${icoPath.pathname} (${ico.length} bytes, sizes: ${sizes.join(", ")})`);

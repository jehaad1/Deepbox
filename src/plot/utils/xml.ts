/**
 * XML helpers for the SVG renderer.
 * @internal
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

function entityFor(code: number): string | undefined {
  switch (code) {
    case 0x26:
      return "&amp;";
    case 0x3c:
      return "&lt;";
    case 0x3e:
      return "&gt;";
    case 0x22:
      return "&quot;";
    case 0x27:
      return "&apos;";
    default:
      return undefined;
  }
}

/** True for C0 controls other than tab, LF and CR, and for U+FFFE/U+FFFF (illegal in XML 1.0). */
function isIllegalXmlChar(code: number): boolean {
  return (
    (code < 0x20 && code !== 0x09 && code !== 0x0a && code !== 0x0d) ||
    code === 0xfffe ||
    code === 0xffff
  );
}

/**
 * Escapes a string for use in XML text or attribute values.
 *
 * The five predefined entities are escaped. Characters that are not allowed in XML 1.0 are
 * removed (control characters) or replaced with U+FFFD (unpaired surrogates), so a stray
 * character in a label can never make the whole SVG document unparseable.
 * @internal
 */
export function escapeXml(s: string): string {
  let out = "";
  let last = 0;
  const n = s.length;
  for (let i = 0; i < n; i++) {
    const code = s.charCodeAt(i);
    let replacement: string | undefined = entityFor(code);
    if (replacement === undefined) {
      if (isIllegalXmlChar(code)) {
        replacement = "";
      } else if (code >= 0xd800 && code <= 0xdbff) {
        const next = i + 1 < n ? s.charCodeAt(i + 1) : 0;
        if (next >= 0xdc00 && next <= 0xdfff) {
          i++; // valid surrogate pair: keep both halves
          continue;
        }
        replacement = "\uFFFD";
      } else if (code >= 0xdc00 && code <= 0xdfff) {
        replacement = "\uFFFD";
      } else {
        continue;
      }
    }
    out += s.slice(last, i) + replacement;
    last = i + 1;
  }
  return last === 0 ? s : out + s.slice(last);
}

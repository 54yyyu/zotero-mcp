// Sample diagrams as an agent following the drawing guide would write them: the fake host's "draw" answer, the
// unit tests and the in-Zotero snapshot use them to judge the palette. One literal hex slips in on purpose
// (the causal graph's #888 edges), as models do; it must come out as a palette colour.

export const PIPELINE = `<svg viewBox="0 0 360 132" xmlns="http://www.w3.org/2000/svg">
  <title>From question to cited answer</title>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M1.5 1.5 8 5 1.5 8.5" fill="none" stroke="muted" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <g stroke-width="1.5">
    <rect x="16" y="30" width="96" height="56" rx="8" fill="surface" stroke="line"/>
    <rect x="132" y="30" width="96" height="56" rx="8" fill="accent-soft" stroke="accent"/>
    <rect x="248" y="30" width="96" height="56" rx="8" fill="surface" stroke="line"/>
    <path d="M114 58h14" stroke="muted" fill="none" marker-end="url(#arrow)"/>
    <path d="M230 58h14" stroke="muted" fill="none" marker-end="url(#arrow)"/>
  </g>
  <g text-anchor="middle" font-size="12">
    <text x="64" y="55" font-weight="600" fill="ink">Search</text>
    <text x="64" y="72" font-size="11" fill="muted">your library</text>
    <text x="180" y="55" font-weight="600" fill="accent">Read</text>
    <text x="180" y="72" font-size="11" fill="muted">the PDF pages</text>
    <text x="296" y="55" font-weight="600" fill="ink">Cite</text>
    <text x="296" y="72" font-size="11" fill="muted">page links</text>
    <text x="180" y="114" font-size="11" fill="muted">every claim opens the page it came from</text>
  </g>
</svg>`;

export const MATRIX = `<svg viewBox="0 0 360 276" xmlns="http://www.w3.org/2000/svg">
  <title>Designs by control and external validity</title>
  <defs>
    <marker id="ax" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M1.5 1.5 8 5 1.5 8.5" fill="none" stroke="muted" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <g stroke-width="1.5">
    <rect x="56" y="12" width="139" height="104" rx="8" fill="teal-soft" stroke="teal"/>
    <rect x="201" y="12" width="139" height="104" rx="8" fill="accent-soft" stroke="accent"/>
    <rect x="56" y="122" width="139" height="104" rx="8" fill="surface" stroke="line"/>
    <rect x="201" y="122" width="139" height="104" rx="8" fill="violet-soft" stroke="violet"/>
    <path d="M42 226V18" stroke="muted" fill="none" marker-end="url(#ax)"/>
    <path d="M56 240h278" stroke="muted" fill="none" marker-end="url(#ax)"/>
  </g>
  <g text-anchor="middle" font-size="12">
    <text x="125.5" y="61" font-weight="600" fill="teal">Natural experiment</text>
    <text x="125.5" y="78" font-size="11" fill="muted">policy shocks</text>
    <text x="270.5" y="61" font-weight="600" fill="accent">Field experiment</text>
    <text x="270.5" y="78" font-size="11" fill="muted">audit studies</text>
    <text x="125.5" y="171" font-weight="600" fill="ink">Observational</text>
    <text x="125.5" y="188" font-size="11" fill="muted">survey regressions</text>
    <text x="270.5" y="171" font-weight="600" fill="violet">Lab experiment</text>
    <text x="270.5" y="188" font-size="11" fill="muted">vignette studies</text>
    <text x="198" y="262" font-size="11" fill="muted">control over who is treated</text>
    <text transform="translate(26 119) rotate(-90)" font-size="11" fill="muted">external validity</text>
  </g>
</svg>`;

export const CAUSAL = `<svg viewBox="0 0 360 196" xmlns="http://www.w3.org/2000/svg">
  <title>Instrument, treatment, outcome</title>
  <defs>
    <marker id="ink" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1 9 5 1 9z" fill="ink"/></marker>
    <marker id="acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1 9 5 1 9z" fill="accent"/></marker>
    <marker id="dim" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1 9 5 1 9z" fill="#888888"/></marker>
  </defs>
  <g stroke-width="1.5" fill="none">
    <path d="M94 136h28" stroke="ink" marker-end="url(#ink)"/>
    <path d="M214 136h36" stroke="accent" stroke-width="2" marker-end="url(#acc)"/>
    <path d="M216 66 189 114" stroke="#888888" stroke-dasharray="4 4" marker-end="url(#dim)"/>
    <path d="M254 66 281 114" stroke="#888888" stroke-dasharray="4 4" marker-end="url(#dim)"/>
  </g>
  <g stroke-width="1.5">
    <rect x="8" y="119" width="84" height="34" rx="17" fill="surface" stroke="line"/>
    <rect x="128" y="119" width="84" height="34" rx="17" fill="accent-soft" stroke="accent"/>
    <rect x="258" y="119" width="94" height="34" rx="17" fill="green-soft" stroke="green"/>
    <rect x="193" y="32" width="84" height="34" rx="17" fill="none" stroke="muted" stroke-dasharray="4 3"/>
  </g>
  <g text-anchor="middle" font-size="12">
    <text x="50" y="140" fill="ink">Distance</text>
    <text x="170" y="140" font-weight="600" fill="accent">Schooling</text>
    <text x="305" y="140" font-weight="600" fill="green">Earnings</text>
    <text x="235" y="53" fill="muted">Ability</text>
    <text x="235" y="20" font-size="11" fill="muted">unobserved</text>
    <text x="180" y="184" font-size="11" fill="muted">distance moves earnings only through schooling</text>
  </g>
</svg>`;

/** Everything a hostile page could try to slip into a drawing. Rendered, it must be an inert circle and text. */
export const HOSTILE = `<?xml version="1.0"?>
<!DOCTYPE svg [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>
<svg viewBox="0 0 200 80" onload="window.__pwned=1" xmlns:xlink="http://www.w3.org/1999/xlink">
  <script>window.__pwned=2</script>
  <style>circle{fill:url(https://evil.example/x.svg)}</style>
  <foreignObject width="200" height="80"><div xmlns="http://www.w3.org/1999/xhtml"><img src="x" onerror="window.__pwned=3"/></div></foreignObject>
  <a href="javascript:window.__pwned=4"><text x="10" y="70">click me</text></a>
  <image href="https://evil.example/track.png" width="10" height="10"/>
  <animate attributeName="href" to="javascript:alert(1)"/>
  <circle cx="40" cy="40" r="20" fill="accent" onclick="window.__pwned=5" style="stroke:red; behavior:url(x.htc)" filter="url(https://evil.example/f)"/>
  <use href="https://evil.example/s.svg#x"/>
  <rect x="80" y="20" width="40" height="40" class="iconbtn" fill="url(javascript:1)"/>
  <text x="140" y="45" font-family="Comic Sans MS">&b;safe</text>
</svg>`;

export const DIAGRAM_ANSWER = `Three ways to picture it.

How the panel answers:

\`\`\`svg
${PIPELINE}
\`\`\`

Where the designs in your library sit:

\`\`\`svg
${MATRIX}
\`\`\`

And the identification argument of the schooling papers:

\`\`\`svg
${CAUSAL}
\`\`\`

The dashed node is not measured, which is why the instrument matters.`;

(function () {
  'use strict';

  // ---------------------------------------------------------------- config

  const MAX_DPR = 2;
  const GRID_EXTENT = 6;
  const BASE_POINT_SIZE = 2.7;
  const CAMERA_DEFAULTS = { yaw: -0.65, pitch: 0.3, distance: 15, target: [0, 0.4, 0] };
  const SYNC_SEARCH_LIMIT = 4000; // below this many words, search on the main thread

  const PALETTE = {
    query: '#78e0ff',
    negative: '#ff8db3',
    neighbor: '#ffc66d',
    neighborFar: '#8f7a55',
    ghost: '#ffe9c4',
    waypoint: '#b7a6e0',
    base: '#7f8ea3',
    axisX: '#ff8db3',
    axisY: '#8de28f',
    axisZ: '#78e0ff',
    grid: '#243549',
    label: 'rgba(6, 12, 21, 0.9)',
  };

  // ---------------------------------------------------------------- dom

  const dom = {
    stage: document.getElementById('stage'),
    datasetSelect: document.getElementById('datasetSelect'),
    starCount: document.getElementById('starCount'),
    glCanvas: document.getElementById('glCanvas'),
    overlay: document.getElementById('overlayCanvas'),
    modeChip: document.getElementById('modeChip'),
    statusBar: document.getElementById('statusBar'),
    resultSummary: document.getElementById('resultSummary'),
    resultTitle: document.getElementById('resultTitle'),
    resultSubtitle: document.getElementById('resultSubtitle'),
    resultList: document.getElementById('resultList'),
    queryChips: document.getElementById('queryChips'),
    queryInput: document.getElementById('queryInput'),
    runButton: document.getElementById('runButton'),
    suggestions: document.getElementById('suggestions'),
    exampleChips: document.getElementById('exampleChips'),
    clearButton: document.getElementById('clearButton'),
    resetButton: document.getElementById('resetButton'),
    topNInput: document.getElementById('topNInput'),
    topNValue: document.getElementById('topNValue'),
    sizeInput: document.getElementById('sizeInput'),
    sizeValue: document.getElementById('sizeValue'),
    fadeInput: document.getElementById('fadeInput'),
    fadeValue: document.getElementById('fadeValue'),
    limitInput: document.getElementById('limitInput'),
    limitNumber: document.getElementById('limitNumber'),
    jumper: document.getElementById('jumper'),
    jumpChips: document.getElementById('jumpChips'),
    jumpSlider: document.getElementById('jumpSlider'),
    dimInput: document.getElementById('dimInput'),
    dimValue: document.getElementById('dimValue'),
    displayMode: document.getElementById('displayMode'),
    pathToggle: document.getElementById('pathToggle'),
    wpInput: document.getElementById('wpInput'),
    wpValue: document.getElementById('wpValue'),
    linesToggle: document.getElementById('linesToggle'),
    labelsToggle: document.getElementById('labelsToggle'),
    rotateToggle: document.getElementById('rotateToggle'),
    animToggle: document.getElementById('animToggle'),
  };

  const octx = dom.overlay.getContext('2d');

  // ---------------------------------------------------------------- state

  const state = {
    // dataset
    words: [],
    count: 0,
    dim: 0,
    embNorm: null,        // Float32Array count*dim, each row unit length
    positions: null,      // Float32Array count*3
    baseColors: null,     // Float32Array count*3 (0..1)
    wordIndex: new Map(),
    mode: 'loading',

    // view options
    topN: 10,
    pointScale: 0.3,
    depthFade: 0.8,      // opacity decay with distance (0 off, 1 aggressive)
    renderLimit: 0,       // words rendered/searched (frequency-ordered prefix)
    dimming: 0.8,         // unselected points: 0 = fully shown, 1 = hidden
    animEnabled: true,
    animT: 1.75,          // selection animation clock (ANIM_TOTAL = idle/complete)
    animDir: 0,           // 1 = playing in, -1 = fading out, 0 = idle
    prevMix: 0,           // outgoing selection cross-fade (1 → 0, unsequenced)
    displayVectors: false, // highlight display: points | vectors from origin
    showPath: true,        // traversal path for expressions
    wpCount: 2,            // closest words shown per waypoint (0-3)
    showLines: true,
    showLabels: true,
    autoRotate: false,

    // selection
    selection: null,
    lastQueryText: '',
    queryToken: 0,

    // interaction
    hoverIndex: -1,
    hoverMark: null,      // {type:'ghost'} | {type:'wp', index} — synthetic markers
    pulseIndex: -1,

    // canvas
    cssWidth: 0,
    cssHeight: 0,
    dpr: 1,

    needsRender: true,
  };

  const camera = {
    yaw: CAMERA_DEFAULTS.yaw,
    pitch: CAMERA_DEFAULTS.pitch,
    distance: CAMERA_DEFAULTS.distance,
    distanceTarget: CAMERA_DEFAULTS.distance,
    target: CAMERA_DEFAULTS.target.slice(),
    velYaw: 0,
    velPitch: 0,
    tween: null,
  };

  const pointerState = {
    dragging: false,
    mode: 'orbit',
    lastX: 0,
    lastY: 0,
    downX: 0,
    downY: 0,
    moved: 0,
    lastMoveTime: 0,
    pointers: new Map(),
    pinchDist: 0,
  };

  // matrices / scratch
  const projMatrix = new Float32Array(16);
  const viewMatrix = new Float32Array(16);
  const viewProj = new Float32Array(16);
  const eye = [0, 0, 1];
  let screenXY = null; // Float32Array count*3: cssX, cssY, clipW (w<=0 -> behind camera)
  let styleArr = null; // Float32Array count*6: r,g,b,a,size,ring

  // ---------------------------------------------------------------- small helpers

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const lerpRgb = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];

  // layered selection animation: dim+query (A) → lines travel (B) → neighbors light (C)
  const DIM_DUR = 1.0;
  const LINE_DUR = 0.5;
  const NEIGH_DUR = 0.25;
  const LINE_START = DIM_DUR;
  const NEIGH_START = DIM_DUR + LINE_DUR;
  const ANIM_TOTAL = DIM_DUR + LINE_DUR + NEIGH_DUR;
  const easeF = (x) => {
    x = clamp(x, 0, 1);
    return x * x * (3 - 2 * x);
  };
  const dimFactor = () => easeF(state.animT / DIM_DUR);
  const lineFactor = () => easeF((state.animT - LINE_START) / LINE_DUR);
  const neighborFactor = () => easeF((state.animT - NEIGH_START) / NEIGH_DUR);

  function hexToRgb(hex) {
    return [
      parseInt(hex.slice(1, 3), 16) / 255,
      parseInt(hex.slice(3, 5), 16) / 255,
      parseInt(hex.slice(5, 7), 16) / 255,
    ];
  }

  function rgbToCss(r, g, b, a) {
    return `rgba(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)}, ${a})`;
  }

  function hexToCss(hex, alpha) {
    const [r, g, b] = hexToRgb(hex);
    return rgbToCss(r, g, b, alpha);
  }

  // ---------------------------------------------------------------- mat4 (column major)

  function mat4Perspective(out, fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    out.fill(0);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far + near) / (near - far);
    out[11] = -1;
    out[14] = (2 * far * near) / (near - far);
    return out;
  }

  function mat4LookAt(out, eyeV, center, up) {
    let zx = eyeV[0] - center[0];
    let zy = eyeV[1] - center[1];
    let zz = eyeV[2] - center[2];
    let len = Math.hypot(zx, zy, zz) || 1;
    zx /= len; zy /= len; zz /= len;
    let xx = up[1] * zz - up[2] * zy;
    let xy = up[2] * zx - up[0] * zz;
    let xz = up[0] * zy - up[1] * zx;
    len = Math.hypot(xx, xy, xz) || 1;
    xx /= len; xy /= len; xz /= len;
    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;
    out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
    out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
    out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
    out[12] = -(xx * eyeV[0] + xy * eyeV[1] + xz * eyeV[2]);
    out[13] = -(yx * eyeV[0] + yy * eyeV[1] + yz * eyeV[2]);
    out[14] = -(zx * eyeV[0] + zy * eyeV[1] + zz * eyeV[2]);
    out[15] = 1;
    return out;
  }

  function mat4Multiply(out, a, b) {
    for (let c = 0; c < 4; c++) {
      const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
      out[c * 4] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
      out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
      out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
      out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
    }
    return out;
  }

  const scratchPoint = { x: 0, y: 0, w: 0 };

  function projectToScreen(x, y, z, out) {
    const m = viewProj;
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (w < 0.02) {
      out.w = -1;
      return false;
    }
    const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
    const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
    out.x = (cx / w * 0.5 + 0.5) * state.cssWidth;
    out.y = (0.5 - cy / w * 0.5) * state.cssHeight;
    out.w = w;
    return true;
  }

  // ---------------------------------------------------------------- expression parser

  const IDENT_START = /[\p{L}_]/u;
  const IDENT_PART = /[\p{L}\p{N}_']/u;

  function buildExpressionError(message, index = null) {
    const error = new Error(index === null ? message : `${message} (column ${index + 1})`);
    error.index = index;
    return error;
  }

  function tokenizeExpression(source) {
    const tokens = [];
    let position = 0;

    while (position < source.length) {
      const character = source[position];

      if (/\s/.test(character)) {
        position++;
        continue;
      }

      if ('+-*()'.includes(character)) {
        tokens.push({ type: 'operator', value: character, index: position });
        position++;
        continue;
      }

      // quoted word: matches any vocabulary token verbatim, including
      // punctuation like "," or ''' and digit words like "1990". Three
      // delimiters so quote-tokens themselves stay expressible: `"` '`'
      if (character === '"' || character === "'" || character === '`') {
        let end = position + 1;
        while (end < source.length && source[end] !== character) end++;
        if (end >= source.length) {
          throw buildExpressionError('Missing closing quote', position);
        }
        const value = source.slice(position + 1, end).toLowerCase();
        if (!value) {
          throw buildExpressionError('Empty quoted word', position);
        }
        tokens.push({ type: 'identifier', value, index: position });
        position = end + 1;
        continue;
      }

      if (/[0-9.]/.test(character)) {
        const start = position;
        let sawDot = false;
        while (position < source.length && /[0-9.]/.test(source[position])) {
          if (source[position] === '.') {
            if (sawDot) break;
            sawDot = true;
          }
          position++;
        }
        const raw = source.slice(start, position);
        if (raw === '.' || Number.isNaN(Number(raw))) {
          throw buildExpressionError(`Invalid number "${raw}"`, start);
        }
        tokens.push({ type: 'number', value: Number(raw), index: start });
        continue;
      }

      if (IDENT_START.test(character)) {
        const start = position;
        // greedy pass: also allow internal - and . so vocabulary tokens like
        // "so-called" or "u.s." parse as one word instead of an expression
        const EXT_PART = /[\p{L}\p{N}_'.\-]/u;
        let extEnd = position;
        while (extEnd < source.length && EXT_PART.test(source[extEnd])) {
          extEnd++;
        }
        const extended = source.slice(start, extEnd).toLowerCase();
        if (extEnd > position && state.wordIndex.has(extended)) {
          tokens.push({ type: 'identifier', value: extended, index: start });
          position = extEnd;
          continue;
        }
        while (position < source.length && IDENT_PART.test(source[position])) {
          position++;
        }
        tokens.push({ type: 'identifier', value: source.slice(start, position).toLowerCase(), index: start });
        continue;
      }

      throw buildExpressionError(`Unexpected character "${character}"`, position);
    }

    return tokens;
  }

  function parseExpressionSource(source) {
    const tokens = tokenizeExpression(source);
    let cursor = 0;

    const peek = () => tokens[cursor] || null;
    const consume = () => tokens[cursor++];

    function matchOperator(...values) {
      const token = peek();
      if (token && token.type === 'operator' && values.includes(token.value)) {
        cursor++;
        return token;
      }
      return null;
    }

    function parsePrimary() {
      const token = peek();
      if (!token) throw buildExpressionError('Unexpected end of expression');

      if (token.type === 'number') {
        consume();
        return { type: 'number', value: token.value, index: token.index };
      }
      if (token.type === 'identifier') {
        consume();
        return { type: 'word', value: token.value, index: token.index };
      }
      if (token.type === 'operator' && token.value === '(') {
        consume();
        const expression = parseSum();
        if (!matchOperator(')')) {
          throw buildExpressionError('Missing closing ")"', token.index);
        }
        return expression;
      }
      throw buildExpressionError(`Unexpected token "${token.value}"`, token.index);
    }

    function parseUnary() {
      const token = peek();
      if (token && token.type === 'operator' && token.value === '-') {
        consume();
        return { type: 'unary', operator: '-', argument: parseUnary(), index: token.index };
      }
      return parsePrimary();
    }

    function parseProduct() {
      let node = parseUnary();
      while (matchOperator('*')) {
        node = { type: 'binary', operator: '*', left: node, right: parseUnary() };
      }
      return node;
    }

    function parseSum() {
      let node = parseProduct();
      while (true) {
        const token = peek();
        if (!token || token.type !== 'operator' || (token.value !== '+' && token.value !== '-')) break;
        consume();
        node = { type: 'binary', operator: token.value, left: node, right: parseProduct() };
      }
      return node;
    }

    const ast = parseSum();
    if (cursor < tokens.length) {
      const token = tokens[cursor];
      throw buildExpressionError(`Unexpected token "${token.value}"`, token.index);
    }
    return ast;
  }

  function collectSignedWords(node, sign = 1, positive = [], negative = []) {
    if (!node) return { positive, negative };
    if (node.type === 'word') {
      (sign >= 0 ? positive : negative).push(node.value);
      return { positive, negative };
    }
    if (node.type === 'unary') {
      return collectSignedWords(node.argument, -sign, positive, negative);
    }
    if (node.type === 'binary') {
      collectSignedWords(node.left, sign, positive, negative);
      collectSignedWords(node.right, node.operator === '-' ? -sign : sign, positive, negative);
    }
    return { positive, negative };
  }

  // Flatten an expression into ordered weighted word terms (for the traversal
  // path). Returns null when the expression is not a plain weighted sum
  // (e.g. word * word).
  function pureNumber(node) {
    if (node.type === 'number') return node.value;
    if (node.type === 'unary') {
      const value = pureNumber(node.argument);
      return value === null ? null : -value;
    }
    if (node.type === 'binary') {
      const left = pureNumber(node.left);
      const right = pureNumber(node.right);
      if (left === null || right === null) return null;
      if (node.operator === '+') return left + right;
      if (node.operator === '-') return left - right;
      return left * right;
    }
    return null;
  }

  function linearTerms(node, scale = 1) {
    if (node.type === 'word') return [{ word: node.value, weight: scale }];
    if (node.type === 'unary') return linearTerms(node.argument, -scale);
    if (node.type === 'binary') {
      if (node.operator === '+' || node.operator === '-') {
        const left = linearTerms(node.left, scale);
        const right = linearTerms(node.right, node.operator === '-' ? -scale : scale);
        return left && right ? [...left, ...right] : null;
      }
      const leftNum = pureNumber(node.left);
      if (leftNum !== null) return linearTerms(node.right, scale * leftNum);
      const rightNum = pureNumber(node.right);
      if (rightNum !== null) return linearTerms(node.left, scale * rightNum);
      return null;
    }
    return null; // bare number term
  }

  function formatTerm(term, isFirst) {
    const magnitude = Math.abs(term.weight);
    const weightText = magnitude === 1 ? '' : `${Number(magnitude.toFixed(2))}·`;
    const sign = isFirst ? (term.weight < 0 ? '−' : '') : (term.weight < 0 ? ' − ' : ' + ');
    return `${sign}${weightText}${term.word}`;
  }

  function getVector(index) {
    return state.embNorm.subarray(index * state.dim, index * state.dim + state.dim);
  }

  function evaluateExpressionNode(node) {
    if (node.type === 'number') {
      return { kind: 'scalar', value: node.value };
    }

    if (node.type === 'word') {
      const index = lookupWord(node.value);
      if (index === undefined) {
        throw buildExpressionError(unknownWordMessage(node.value), node.index);
      }
      return { kind: 'vector', value: new Float32Array(getVector(index)) };
    }

    if (node.type === 'unary') {
      const value = evaluateExpressionNode(node.argument);
      if (value.kind === 'scalar') {
        return { kind: 'scalar', value: -value.value };
      }
      const vector = value.value;
      for (let i = 0; i < vector.length; i++) vector[i] = -vector[i];
      return { kind: 'vector', value: vector };
    }

    if (node.type === 'binary') {
      const left = evaluateExpressionNode(node.left);
      const right = evaluateExpressionNode(node.right);

      if (node.operator === '+' || node.operator === '-') {
        if (left.kind !== right.kind) {
          throw buildExpressionError('Addition and subtraction need two words or two numbers');
        }
        if (left.kind === 'scalar') {
          return { kind: 'scalar', value: node.operator === '+' ? left.value + right.value : left.value - right.value };
        }
        const vector = left.value;
        const other = right.value;
        for (let i = 0; i < vector.length; i++) {
          vector[i] = node.operator === '+' ? vector[i] + other[i] : vector[i] - other[i];
        }
        return { kind: 'vector', value: vector };
      }

      // multiplication
      if (left.kind === 'scalar' && right.kind === 'scalar') {
        return { kind: 'scalar', value: left.value * right.value };
      }
      if (left.kind !== right.kind) {
        const vector = left.kind === 'vector' ? left.value : right.value;
        const scalar = left.kind === 'scalar' ? left.value : right.value;
        for (let i = 0; i < vector.length; i++) vector[i] *= scalar;
        return { kind: 'vector', value: vector };
      }
      const vector = left.value;
      for (let i = 0; i < vector.length; i++) vector[i] *= right.value[i];
      return { kind: 'vector', value: vector };
    }

    throw buildExpressionError('Could not evaluate expression');
  }

  // A word only "exists" within the current render/search limit — everything
  // (search, arithmetic, autocomplete) sees the same frequency-ordered prefix.
  function lookupWord(word) {
    const index = state.wordIndex.get(word);
    return index !== undefined && index < visibleCount() ? index : undefined;
  }

  function unknownWordMessage(word) {
    const raw = state.wordIndex.get(word);
    if (raw !== undefined && raw >= visibleCount()) {
      return `"${word}" is word #${(raw + 1).toLocaleString()} — raise "Words shown" to include it`;
    }
    const guess = suggestClosestWord(word);
    return guess
      ? `"${word}" is not in this dataset — did you mean "${guess}"?`
      : `"${word}" is not in this dataset`;
  }

  function suggestClosestWord(word) {
    let best = null;
    let bestDist = 3;
    const limit = visibleCount();
    for (let i = 0; i < limit; i++) {
      const candidate = state.words[i];
      if (Math.abs(candidate.length - word.length) > 2) continue;
      if (candidate[0] !== word[0]) continue;
      const d = editDistance(word, candidate, bestDist);
      if (d < bestDist) {
        bestDist = d;
        best = candidate;
        if (d === 1) break;
      }
    }
    return best;
  }

  function editDistance(a, b, cap) {
    const la = a.length, lb = b.length;
    let prev = new Array(lb + 1);
    let curr = new Array(lb + 1);
    for (let j = 0; j <= lb; j++) prev[j] = j;
    for (let i = 1; i <= la; i++) {
      curr[0] = i;
      let rowMin = i;
      for (let j = 1; j <= lb; j++) {
        curr[j] = Math.min(
          prev[j] + 1,
          curr[j - 1] + 1,
          prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
        );
        if (curr[j] < rowMin) rowMin = curr[j];
      }
      if (rowMin > cap) return cap + 1;
      [prev, curr] = [curr, prev];
    }
    return prev[lb];
  }

  // ---------------------------------------------------------------- nearest-neighbor search

  function topNeighborsSync(query, topN, excludeSet, limit) {
    const { embNorm, dim } = state;
    const count = limit;
    const idx = new Int32Array(topN).fill(-1);
    const sc = new Float32Array(topN).fill(-2);

    for (let i = 0; i < count; i++) {
      if (excludeSet.has(i)) continue;
      let dot = 0;
      const offset = i * dim;
      for (let j = 0; j < dim; j++) dot += embNorm[offset + j] * query[j];
      if (dot > sc[topN - 1]) {
        let p = topN - 1;
        while (p > 0 && sc[p - 1] < dot) {
          sc[p] = sc[p - 1];
          idx[p] = idx[p - 1];
          p--;
        }
        sc[p] = dot;
        idx[p] = i;
      }
    }

    const neighbors = [];
    for (let i = 0; i < topN; i++) {
      if (idx[i] >= 0) neighbors.push({ index: idx[i], score: sc[i] });
    }
    return neighbors;
  }

  const WORKER_SOURCE = `
    let emb = null, count = 0, dim = 0;
    onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'init') {
        emb = new Float32Array(msg.buffer);
        count = msg.count;
        dim = msg.dim;
        postMessage({ type: 'ready' });
        return;
      }
      if (msg.type === 'query') {
        const q = new Float32Array(msg.vector);
        const topN = msg.topN;
        const exclude = new Set(msg.exclude || []);
        const limit = Math.min(count, msg.limit || count);
        const idx = new Int32Array(topN).fill(-1);
        const sc = new Float32Array(topN).fill(-2);
        for (let i = 0; i < limit; i++) {
          if (exclude.has(i)) continue;
          let dot = 0;
          const offset = i * dim;
          for (let j = 0; j < dim; j++) dot += emb[offset + j] * q[j];
          if (dot > sc[topN - 1]) {
            let p = topN - 1;
            while (p > 0 && sc[p - 1] < dot) {
              sc[p] = sc[p - 1];
              idx[p] = idx[p - 1];
              p--;
            }
            sc[p] = dot;
            idx[p] = i;
          }
        }
        const neighbors = [];
        for (let i = 0; i < topN; i++) {
          if (idx[i] >= 0) neighbors.push({ index: idx[i], score: sc[i] });
        }
        postMessage({ type: 'result', id: msg.id, neighbors });
      }
    };
  `;

  const search = {
    worker: null,
    ready: false,
    nextId: 1,
    pending: new Map(),
  };

  function initSearchWorker() {
    if (search.worker) {
      search.worker.terminate();
      search.worker = null;
    }
    search.ready = false;
    search.pending.clear();
    if (state.count <= SYNC_SEARCH_LIMIT) return;
    try {
      const blob = new Blob([WORKER_SOURCE], { type: 'application/javascript' });
      const worker = new Worker(URL.createObjectURL(blob));
      worker.onmessage = (event) => {
        const msg = event.data;
        if (msg.type === 'ready') {
          search.ready = true;
          return;
        }
        if (msg.type === 'result') {
          const resolve = search.pending.get(msg.id);
          if (resolve) {
            search.pending.delete(msg.id);
            resolve(msg.neighbors);
          }
        }
      };
      worker.onerror = () => {
        search.worker = null;
        search.ready = false;
      };
      const copy = state.embNorm.slice();
      worker.postMessage({ type: 'init', buffer: copy.buffer, count: state.count, dim: state.dim }, [copy.buffer]);
      search.worker = worker;
    } catch (error) {
      search.worker = null;
    }
  }

  function searchNeighbors(vector, topN, excludeSet) {
    // normalize a copy of the query so dot product == cosine
    const query = new Float32Array(vector);
    let norm = 0;
    for (let i = 0; i < query.length; i++) norm += query[i] * query[i];
    norm = Math.sqrt(norm);
    if (norm < 1e-6) {
      return Promise.reject(buildExpressionError('The expression cancels out to a zero vector'));
    }
    for (let i = 0; i < query.length; i++) query[i] /= norm;

    const limit = visibleCount();

    if (search.worker && search.ready) {
      return new Promise((resolve) => {
        const id = search.nextId++;
        search.pending.set(id, resolve);
        search.worker.postMessage(
          { type: 'query', id, vector: query.buffer, topN, exclude: [...excludeSet], limit },
          [query.buffer]
        );
      });
    }

    return Promise.resolve(topNeighborsSync(query, topN, excludeSet, limit));
  }

  // ---------------------------------------------------------------- data loading

  function finalizeDataset() {
    state.wordIndex = new Map(state.words.map((word, index) => [word, index]));
    normalizeEmbeddings();
    normalizeProjection();
    screenXY = new Float32Array(state.count * 3);
    styleArr = new Float32Array(state.count * 6);
    renderer.uploadPositions();
    initSearchWorker();
    applySelectionStyles();
    state.needsRender = true;
    dom.topNInput.max = String(Math.min(50, Math.max(4, state.count - 1)));

    state.renderLimit = state.count;
    const minLimit = Math.min(100, state.count);
    dom.limitInput.min = String(minLimit);
    dom.limitInput.max = String(state.count);
    dom.limitInput.step = String(state.count > 2000 ? 100 : 1);
    dom.limitInput.value = String(state.count);
    dom.limitNumber.min = String(minLimit);
    dom.limitNumber.max = String(state.count);
    dom.limitNumber.value = String(state.count);
  }

  function visibleCount() {
    return Math.min(state.count, state.renderLimit || state.count);
  }

  function normalizeEmbeddings() {
    const { embNorm, count, dim } = state;
    for (let i = 0; i < count; i++) {
      const offset = i * dim;
      let norm = 0;
      for (let j = 0; j < dim; j++) norm += embNorm[offset + j] * embNorm[offset + j];
      norm = Math.sqrt(norm) || 1;
      for (let j = 0; j < dim; j++) embNorm[offset + j] /= norm;
    }
  }

  function normalizeProjection() {
    const pos = state.positions;
    const count = state.count;
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < count; i++) {
      cx += pos[i * 3];
      cy += pos[i * 3 + 1];
      cz += pos[i * 3 + 2];
    }
    cx /= count; cy /= count; cz /= count;

    const radii = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const x = pos[i * 3] - cx;
      const y = pos[i * 3 + 1] - cy;
      const z = pos[i * 3 + 2] - cz;
      radii[i] = Math.hypot(x, y, z);
    }
    const sorted = radii.slice().sort();
    const p95 = sorted[Math.floor(count * 0.95)] || 1;
    const scale = 4.4 / p95;
    state.cloudRadius = 4.4; // p95 radius after rescaling below
    // always recenter + rescale: tier prefixes of a large UMAP run have
    // arbitrary centroids, so each loaded slice must be normalized to the grid
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (pos[i * 3] - cx) * scale;
      pos[i * 3 + 1] = (pos[i * 3 + 1] - cy) * scale;
      pos[i * 3 + 2] = (pos[i * 3 + 2] - cz) * scale;
    }
  }

  function colorFromPosition(x, y, z) {
    // saturated hues so points read against the dark bg and density stacks
    const hue = (Math.atan2(z, x) / (Math.PI * 2) + 0.5) * 360;
    const light = clamp(0.62 + y * 0.03, 0.52, 0.72);
    const [r, g, b] = hslToRgb(hue, 0.52, light);
    const base = hexToRgb(PALETTE.base);
    return [lerp(base[0], r, 0.8), lerp(base[1], g, 0.8), lerp(base[2], b, 0.8)];
  }

  function hslToRgb(h, s, l) {
    const a = s * Math.min(l, 1 - l);
    const f = (n) => {
      const k = (n + h / 30) % 12;
      return l - a * Math.max(-1, Math.min(k - 3, Math.min(9 - k, 1)));
    };
    return [f(0), f(8), f(4)];
  }

  function loadDemoData() {
    const demo = window.EMBEDDING_DEMO;
    state.words = demo.words;
    state.count = demo.count;
    state.dim = demo.vectorSize;
    state.embNorm = new Float32Array(demo.embeddings);
    state.positions = new Float32Array(demo.projected);
    state.baseColors = new Float32Array(state.count * 3);
    for (let i = 0; i < state.count; i++) {
      const [r, g, b] = hexToRgb(demo.baseColors[i]);
      state.baseColors[i * 3] = r;
      state.baseColors[i * 3 + 1] = g;
      state.baseColors[i * 3 + 2] = b;
    }
    state.mode = 'demo';
    dom.modeChip.textContent = `Demo dataset · ${state.count} words`;
    finalizeDataset();
    setStatus(`Demo dataset ready — ${state.count} words. Try "king - man + woman".`);
  }

  async function fetchBinary(url, onProgress) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    const total = Number(response.headers.get('Content-Length')) || 0;
    if (!response.body || !total || !onProgress) {
      return response.arrayBuffer();
    }
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress(received / total);
    }
    const buffer = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.length;
    }
    return buffer.buffer;
  }

  async function loadBinaryFrom(prefix) {
    const manifestResponse = await fetch(`${prefix}manifest.json`);
    if (!manifestResponse.ok) throw new Error('no manifest');
    const manifest = await manifestResponse.json();

    const progress = [0, 0];
    const updateChip = () => {
      const pct = Math.round(((progress[0] + progress[1]) / 2) * 100);
      dom.modeChip.textContent = `Loading vectors… ${pct}%`;
    };

    const [embBuffer, projBuffer] = await Promise.all([
      fetchBinary(`${prefix}embeddings.bin`, (p) => { progress[0] = p; updateChip(); }),
      fetchBinary(`${prefix}projected.bin`, (p) => { progress[1] = p; updateChip(); }),
    ]);

    state.words = manifest.words;
    state.count = manifest.count || manifest.words.length;
    state.dim = manifest.vectorLength || 50;
    state.embNorm = new Float32Array(embBuffer);
    state.positions = new Float32Array(projBuffer);
    state.baseColors = new Float32Array(state.count * 3);
    for (let i = 0; i < state.count; i++) {
      const [r, g, b] = colorFromPosition(
        state.positions[i * 3],
        state.positions[i * 3 + 1],
        state.positions[i * 3 + 2]
      );
      state.baseColors[i * 3] = r;
      state.baseColors[i * 3 + 1] = g;
      state.baseColors[i * 3 + 2] = b;
    }
    state.mode = 'binary';
    dom.modeChip.textContent = `GloVe · ${state.count.toLocaleString()} words`;
    finalizeDataset();
    setStatus(`Loaded ${state.count.toLocaleString()} word vectors.`);
  }

  let datasetLoading = false;

  async function loadSizedDataset(size) {
    if (datasetLoading) return;
    datasetLoading = true;
    dom.datasetSelect.disabled = true;
    try {
      await loadBinaryFrom(`data/${size}/`);
      dom.datasetSelect.value = String(size);
      clearSelection();
    } catch (error) {
      setStatus('Could not load that dataset size.');
    } finally {
      datasetLoading = false;
      dom.datasetSelect.disabled = false;
    }
  }

  function populateDatasetSelect(index) {
    dom.datasetSelect.innerHTML = '';
    for (const size of index.sizes) {
      const option = document.createElement('option');
      option.value = String(size);
      const mb = (size * ((index.vectorLength || 50) * 4 + 20)) / 1e6;
      option.textContent = `${size.toLocaleString()} words · ~${mb < 1 ? mb.toFixed(1) : Math.round(mb)} MB`;
      dom.datasetSelect.appendChild(option);
    }
    dom.datasetSelect.hidden = false;
  }

  async function loadData() {
    if (location.protocol === 'file:') {
      loadDemoData();
      return;
    }

    try {
      const indexResponse = await fetch('data/index.json');
      if (indexResponse.ok) {
        const index = await indexResponse.json();
        populateDatasetSelect(index);
        const size = index.sizes.includes(index.default) ? index.default : index.sizes[0];
        await loadSizedDataset(size);
        if (state.mode === 'binary') return;
      }
    } catch (error) { /* fall through */ }

    try {
      await loadBinaryFrom(''); // legacy single dataset at the repo root
    } catch (error) {
      loadDemoData();
    }
  }

  // ---------------------------------------------------------------- WebGL renderer

  const POINT_VS = `
    attribute vec3 aPos;
    attribute vec4 aColor;
    attribute vec2 aExtra; // size, phase (0 bg, 1 query+ring, 2 neighbor, 3 waypoint word)
    attribute vec3 aBase;  // resting cloud color
    attribute vec4 aPrevColor; // outgoing selection styling, cross-fades away
    attribute vec2 aPrevExtra;
    uniform float uPrevMix;
    uniform mat4 uViewProj;
    uniform vec3 uEye;
    uniform float uDist;
    uniform float uSizeScale;
    uniform float uDimMix;   // phase A: background dims, query lights up
    uniform float uNeighMix; // phase C: neighbors + waypoint words light up
    uniform float uBaseSize;
    uniform float uFade;   // 0 = no depth fade, 1 = back of cloud invisible
    uniform float uRadius; // cloud radius in world units
    varying vec4 vColor;
    varying float vRing;
    varying float vSize;
    varying float vFog;
    void main() {
      vec4 clip = uViewProj * vec4(aPos, 1.0);
      gl_Position = clip;
      float w = max(clip.w, 0.1);
      float phase = aExtra.y;
      float m = phase >= 1.5 ? uNeighMix : uDimMix;
      vec3 rgb = mix(aBase, aColor.rgb, m);
      float alpha = mix(0.52, aColor.a, m);
      float sizePt = mix(uBaseSize, aExtra.x, m);
      // outgoing selection: its highlighted points fade to their new style
      // all at once, in parallel with the incoming animation
      float pf = uPrevMix * step(0.5, aPrevExtra.y);
      rgb = mix(rgb, aPrevColor.rgb, pf);
      alpha = mix(alpha, aPrevColor.a, pf);
      sizePt = mix(sizePt, aPrevExtra.x, pf);
      // fixed reference distance: true perspective — points grow as the
      // camera approaches, regardless of where the orbit target sits
      gl_PointSize = clamp(sizePt * uSizeScale * (15.0 / w), 1.5, 256.0);
      vSize = gl_PointSize;
      // fog band anchored to the cloud itself: t=0 at its nearest reachable
      // depth, t=1 at its farthest — consistent grading at any zoom level
      float d = distance(aPos, uEye);
      float nearB = max(uDist - uRadius, 0.0);
      // hybrid fog: mostly relative (graded across the whole cloud span) with
      // a mild absolute near-field term for immediate depth separation
      float tRel = clamp((d - nearB) / max(2.0 * uRadius, 0.001), 0.0, 1.0);
      float tAbs = clamp((d - nearB) / max(uRadius * 0.8, 0.001), 0.0, 1.0);
      float fogT = uFade * (0.7 * smoothstep(0.05, 0.95, tRel) + 0.3 * smoothstep(0.0, 0.7, tAbs));
      // highlighted points (fully opaque style) mostly resist the fog
      fogT *= mix(1.0, 0.3, step(0.99, alpha));
      vFog = fogT * 0.85;
      vColor = vec4(rgb, alpha * (1.0 - fogT * fogT));
      float newRing = (phase > 0.5 && phase < 1.5) ? uDimMix : 0.0;
      float prevRing = (aPrevExtra.y > 0.5 && aPrevExtra.y < 1.5) ? pf : 0.0;
      vRing = max(newRing, prevRing);
    }
  `;

  const POINT_FS = `
    precision mediump float;
    varying vec4 vColor;
    varying float vRing;
    varying float vSize;
    varying float vFog;
    void main() {
      vec2 uv = gl_PointCoord * 2.0 - 1.0;
      float r = length(uv);
      if (r > 1.0) discard;
      // ~2px antialiased rim regardless of point size — big points stay crisp
      float aa = clamp(2.5 / vSize, 0.02, 0.28);
      float edge = 1.0 - smoothstep(1.0 - aa, 1.0, r);
      // flat disc — no radial brightness gradient, no bloom
      vec3 col = vColor.rgb;
      // atmospheric fog: distant points drift toward the background color
      col = mix(col, vec3(0.033, 0.067, 0.11), vFog);
      float alpha = vColor.a * edge;
      if (vRing > 0.5) {
        float ring = smoothstep(0.5, 0.64, r) * (1.0 - smoothstep(0.8, 0.94, r));
        col = mix(col, vec3(1.0), ring * 0.85);
        alpha = max(alpha, vColor.a * ring * 0.95);
      }
      gl_FragColor = vec4(col * alpha, alpha);
    }
  `;

  const LINE_VS = `
    attribute vec3 aPos;
    attribute vec4 aColor;
    uniform mat4 uViewProj;
    varying vec4 vColor;
    void main() {
      gl_Position = uViewProj * vec4(aPos, 1.0);
      vColor = aColor;
    }
  `;

  const LINE_FS = `
    precision mediump float;
    varying vec4 vColor;
    uniform float uAlpha;
    void main() {
      float a = vColor.a * uAlpha;
      gl_FragColor = vec4(vColor.rgb * a, a);
    }
  `;

  const renderer = {
    gl: null,
    pointProgram: null,
    lineProgram: null,
    posBuffer: null,
    styleBuffer: null,
    gridBuffer: null,
    gridVertexCount: 0,
    dynBuffer: null,
    dynVertexCount: 0,
    loc: {},

    init() {
      const gl = dom.glCanvas.getContext('webgl', {
        antialias: true,
        alpha: true,
        premultipliedAlpha: true,
        powerPreference: 'high-performance',
      });
      if (!gl) {
        setStatus('WebGL is not available in this browser.');
        return;
      }
      this.gl = gl;

      this.pointProgram = createProgram(gl, POINT_VS, POINT_FS);
      this.lineProgram = createProgram(gl, LINE_VS, LINE_FS);

      this.loc = {
        pPos: gl.getAttribLocation(this.pointProgram, 'aPos'),
        pColor: gl.getAttribLocation(this.pointProgram, 'aColor'),
        pExtra: gl.getAttribLocation(this.pointProgram, 'aExtra'),
        pViewProj: gl.getUniformLocation(this.pointProgram, 'uViewProj'),
        pEye: gl.getUniformLocation(this.pointProgram, 'uEye'),
        pDist: gl.getUniformLocation(this.pointProgram, 'uDist'),
        pSizeScale: gl.getUniformLocation(this.pointProgram, 'uSizeScale'),
        pFade: gl.getUniformLocation(this.pointProgram, 'uFade'),
        pRadius: gl.getUniformLocation(this.pointProgram, 'uRadius'),
        pBase: gl.getAttribLocation(this.pointProgram, 'aBase'),
        pDimMix: gl.getUniformLocation(this.pointProgram, 'uDimMix'),
        pNeighMix: gl.getUniformLocation(this.pointProgram, 'uNeighMix'),
        pBaseSize: gl.getUniformLocation(this.pointProgram, 'uBaseSize'),
        lPos: gl.getAttribLocation(this.lineProgram, 'aPos'),
        lColor: gl.getAttribLocation(this.lineProgram, 'aColor'),
        lViewProj: gl.getUniformLocation(this.lineProgram, 'uViewProj'),
        lAlpha: gl.getUniformLocation(this.lineProgram, 'uAlpha'),
        pPrevColor: gl.getAttribLocation(this.pointProgram, 'aPrevColor'),
        pPrevExtra: gl.getAttribLocation(this.pointProgram, 'aPrevExtra'),
        pPrevMix: gl.getUniformLocation(this.pointProgram, 'uPrevMix'),
      };

      this.posBuffer = gl.createBuffer();
      this.baseColorBuffer = gl.createBuffer();
      this.styleBuffer = gl.createBuffer();
      this.prevStyleBuffer = gl.createBuffer();
      this.gridBuffer = gl.createBuffer();
      this.dynBuffer = gl.createBuffer();
      this.prevDynBuffer = gl.createBuffer();
      this.prevDynCount = 0;
      this.lastLineData = null;

      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.clearColor(0, 0, 0, 0);

      this.buildGrid();

      dom.glCanvas.addEventListener('webglcontextlost', (event) => event.preventDefault());
      dom.glCanvas.addEventListener('webglcontextrestored', () => {
        this.init();
        this.uploadPositions();
        applySelectionStyles();
        state.needsRender = true;
      });
    },

    buildGrid() {
      const gl = this.gl;
      const verts = [];
      const gridColor = hexToRgb(PALETTE.grid);

      const pushLine = (x1, y1, z1, x2, y2, z2, rgb, a) => {
        verts.push(x1, y1, z1, rgb[0], rgb[1], rgb[2], a);
        verts.push(x2, y2, z2, rgb[0], rgb[1], rgb[2], a);
      };

      for (let i = -GRID_EXTENT; i <= GRID_EXTENT; i++) {
        if (i === 0) continue;
        pushLine(-GRID_EXTENT, 0, i, GRID_EXTENT, 0, i, gridColor, 0.5);
        pushLine(i, 0, -GRID_EXTENT, i, 0, GRID_EXTENT, gridColor, 0.5);
      }
      pushLine(-GRID_EXTENT, 0, 0, GRID_EXTENT, 0, 0, hexToRgb(PALETTE.axisX), 0.85);
      pushLine(0, -GRID_EXTENT, 0, 0, GRID_EXTENT, 0, hexToRgb(PALETTE.axisY), 0.85);
      pushLine(0, 0, -GRID_EXTENT, 0, 0, GRID_EXTENT, hexToRgb(PALETTE.axisZ), 0.85);

      const data = new Float32Array(verts);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      this.gridVertexCount = data.length / 7;
    },

    uploadPositions() {
      const gl = this.gl;
      if (!gl || !state.positions) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, state.positions, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.baseColorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, state.baseColors, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.styleBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, state.count * 6 * 4, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.prevStyleBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, state.count * 6 * 4, gl.DYNAMIC_DRAW);
      this.prevDynCount = 0;
      this.lastLineData = null;
    },

    // freeze the outgoing selection (point styles + lines) so it can
    // cross-fade away while the incoming selection animates
    snapshotPrev() {
      const gl = this.gl;
      if (!gl || !styleArr) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.prevStyleBuffer);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, styleArr);
      this.prevDynCount = 0;
      if (this.lastLineData && this.lastLineData.length) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.prevDynBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.lastLineData, gl.DYNAMIC_DRAW);
        this.prevDynCount = this.lastLineData.length / 7;
      }
    },

    uploadStyles() {
      const gl = this.gl;
      if (!gl || !styleArr) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.styleBuffer);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, styleArr);
    },

    uploadDynamicLines(data) {
      const gl = this.gl;
      if (!gl) return;
      this.dynVertexCount = data.length / 7;
      this.lastLineData = data;
      if (!data.length) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.dynBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    },

    drawLines(buffer, count, alpha = 1) {
      const gl = this.gl;
      if (!count) return;
      gl.useProgram(this.lineProgram);
      gl.uniformMatrix4fv(this.loc.lViewProj, false, viewProj);
      gl.uniform1f(this.loc.lAlpha, alpha);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.enableVertexAttribArray(this.loc.lPos);
      gl.vertexAttribPointer(this.loc.lPos, 3, gl.FLOAT, false, 28, 0);
      gl.enableVertexAttribArray(this.loc.lColor);
      gl.vertexAttribPointer(this.loc.lColor, 4, gl.FLOAT, false, 28, 12);
      gl.drawArrays(gl.LINES, 0, count);
    },

    draw() {
      const gl = this.gl;
      if (!gl) return;
      gl.viewport(0, 0, dom.glCanvas.width, dom.glCanvas.height);
      gl.clear(gl.COLOR_BUFFER_BIT);

      this.drawLines(this.gridBuffer, this.gridVertexCount);
      if (state.prevMix > 0 && this.prevDynCount) {
        this.drawLines(this.prevDynBuffer, this.prevDynCount, easeF(state.prevMix));
      }
      this.drawLines(this.dynBuffer, this.dynVertexCount);

      if (!state.count) return;

      gl.useProgram(this.pointProgram);
      gl.uniformMatrix4fv(this.loc.pViewProj, false, viewProj);
      gl.uniform3f(this.loc.pEye, eye[0], eye[1], eye[2]);
      gl.uniform1f(this.loc.pDist, camera.distance);
      gl.uniform1f(this.loc.pSizeScale, state.dpr * state.pointScale);
      gl.uniform1f(this.loc.pFade, state.depthFade);
      gl.uniform1f(this.loc.pRadius, state.cloudRadius || 4.4);
      gl.uniform1f(this.loc.pDimMix, dimFactor());
      gl.uniform1f(this.loc.pNeighMix, neighborFactor());
      gl.uniform1f(this.loc.pBaseSize, BASE_POINT_SIZE);
      gl.uniform1f(this.loc.pPrevMix, easeF(state.prevMix));

      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
      gl.enableVertexAttribArray(this.loc.pPos);
      gl.vertexAttribPointer(this.loc.pPos, 3, gl.FLOAT, false, 12, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.baseColorBuffer);
      gl.enableVertexAttribArray(this.loc.pBase);
      gl.vertexAttribPointer(this.loc.pBase, 3, gl.FLOAT, false, 12, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.styleBuffer);
      gl.enableVertexAttribArray(this.loc.pColor);
      gl.vertexAttribPointer(this.loc.pColor, 4, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(this.loc.pExtra);
      gl.vertexAttribPointer(this.loc.pExtra, 2, gl.FLOAT, false, 24, 16);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.prevStyleBuffer);
      gl.enableVertexAttribArray(this.loc.pPrevColor);
      gl.vertexAttribPointer(this.loc.pPrevColor, 4, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(this.loc.pPrevExtra);
      gl.vertexAttribPointer(this.loc.pPrevExtra, 2, gl.FLOAT, false, 24, 16);

      gl.drawArrays(gl.POINTS, 0, visibleCount());
    },
  };

  function createProgram(gl, vsSource, fsSource) {
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader) || 'shader compile failed');
      }
      return shader;
    };
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vsSource));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fsSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) || 'program link failed');
    }
    return program;
  }

  // ---------------------------------------------------------------- selection styling

  function applySelectionStyles() {
    if (!styleArr) return;
    const selection = state.selection;
    const count = state.count;
    const base = state.baseColors;

    if (!selection) {
      for (let i = 0; i < count; i++) {
        const o = i * 6;
        styleArr[o] = base[i * 3];
        styleArr[o + 1] = base[i * 3 + 1];
        styleArr[o + 2] = base[i * 3 + 2];
        styleArr[o + 3] = 0.52; // half-opacity base: overlaps accumulate, so
        styleArr[o + 4] = BASE_POINT_SIZE; // dense regions visibly glow
        styleArr[o + 5] = 0;
      }
      renderer.uploadStyles();
      return;
    }

    const positiveSet = new Set(selection.positive);
    const negativeSet = new Set(selection.negative);
    const neighborRank = new Map();
    selection.neighbors.forEach((n, rank) => neighborRank.set(n.index, rank));

    // waypoint words inherit their stop's position on the start→end gradient
    const wpNeighborColor = new Map();
    if (state.showPath && selection.path && state.wpCount > 0) {
      const wps = selection.path.waypoints;
      const hops = wps.length + 1;
      const startRgb = hexToRgb(PALETTE.query);
      const endRgb = hexToRgb(PALETTE.ghost);
      const gray = hexToRgb(PALETTE.base);
      wps.forEach((wp, wi) => {
        const color = lerpRgb(lerpRgb(startRgb, endRgb, (wi + 1) / hops), gray, 0.25);
        for (const n of wp.neighbors.slice(0, state.wpCount)) {
          if (!wpNeighborColor.has(n.index)) wpNeighborColor.set(n.index, color);
        }
      });
    }

    // perceptual-ish curve: 0% = fully shown, default is a faint ghost,
    // 100% = invisible (also removes them from hover/click picking)
    const othersAlpha = 0.52 * Math.pow(1 - state.dimming, 1.5);
    const queryRgb = hexToRgb(PALETTE.query);
    const negativeRgb = hexToRgb(PALETTE.negative);
    const neighborRgb = hexToRgb(PALETTE.neighbor);
    const neighborFarRgb = hexToRgb(PALETTE.neighborFar);
    const waypointRgb = hexToRgb(PALETTE.waypoint);
    const total = Math.max(1, selection.neighbors.length - 1);

    const grayRgb = hexToRgb(PALETTE.base);
    for (let i = 0; i < count; i++) {
      const o = i * 6;
      // unselected points lose their hue — selection reads as "color returns
      // to the chosen few", not as a background flash
      let r = grayRgb[0], g = grayRgb[1], b = grayRgb[2];
      let alpha = othersAlpha;
      let size = BASE_POINT_SIZE * 0.9;
      let ring = 0;

      const rank = neighborRank.get(i);
      if (positiveSet.has(i)) {
        [r, g, b] = queryRgb;
        alpha = 1;
        size = 4.8;
        ring = 1; // phase 1: lights with the dim wave, carries the ring
      } else if (negativeSet.has(i)) {
        [r, g, b] = negativeRgb;
        alpha = 1;
        size = 4.5;
        ring = 1;
      } else if (rank !== undefined) {
        const t = rank / total;
        r = lerp(neighborRgb[0], neighborFarRgb[0], t * 0.7);
        g = lerp(neighborRgb[1], neighborFarRgb[1], t * 0.7);
        b = lerp(neighborRgb[2], neighborFarRgb[2], t * 0.7);
        alpha = 1;
        size = 4.5 - 1.3 * t;
        ring = 2; // phase 2: lights after the lines arrive
      } else if (wpNeighborColor.has(i)) {
        [r, g, b] = wpNeighborColor.get(i);
        alpha = 0.95;
        size = 3.6;
        ring = 3; // waypoint words light with the neighbors
      }

      styleArr[o] = r;
      styleArr[o + 1] = g;
      styleArr[o + 2] = b;
      styleArr[o + 3] = alpha;
      styleArr[o + 4] = size;
      styleArr[o + 5] = ring;
    }
    renderer.uploadStyles();
  }

  function rebuildDynamicLines(growth = 1) {
    const verts = [];
    const selection = state.selection;

    // grow: lines extend from their anchor toward the endpoint as the
    // selection animation plays
    const pushSeg = (x1, y1, z1, x2, y2, z2, rgb, a, g = growth) => {
      if (g <= 0) return;
      const gx = x1 + (x2 - x1) * g;
      const gy = y1 + (y2 - y1) * g;
      const gz = z1 + (z2 - z1) * g;
      verts.push(x1, y1, z1, rgb[0], rgb[1], rgb[2], a);
      verts.push(gx, gy, gz, rgb[0], rgb[1], rgb[2], a);
    };

    if (selection) {
      const pos = state.positions;
      const queryRgb = hexToRgb(PALETTE.query);
      const negativeRgb = hexToRgb(PALETTE.negative);
      const neighborRgb = hexToRgb(PALETTE.neighbor);
      const ghostRgb = hexToRgb(PALETTE.ghost);
      const waypointRgb = hexToRgb(PALETTE.waypoint);
      const at = (i) => [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];

      if (state.displayVectors) {
        for (const i of selection.positive) {
          const p = at(i);
          pushSeg(0, 0, 0, p[0], p[1], p[2], queryRgb, 0.8);
        }
        for (const i of selection.negative) {
          const p = at(i);
          pushSeg(0, 0, 0, p[0], p[1], p[2], negativeRgb, 0.8);
        }
        if (selection.ghost) {
          pushSeg(0, 0, 0, selection.ghost[0], selection.ghost[1], selection.ghost[2], ghostRgb, 0.9);
        }
      }

      if (state.showLines) {
        const anchor = selection.ghost
          ? selection.ghost
          : selection.queryWord !== null && selection.queryWord !== undefined
            ? at(selection.queryWord)
            : null;
        if (anchor) {
          for (const n of selection.neighbors) {
            const p = at(n.index);
            pushSeg(anchor[0], anchor[1], anchor[2], p[0], p[1], p[2], neighborRgb, 0.22);
          }
        }
      }

      if (state.showPath && selection.path && selection.ghost) {
        const points = [
          at(selection.path.startIndex),
          ...selection.path.waypoints.map((wp) => wp.pos),
          selection.ghost,
        ];
        // path hops grow sequentially — the traversal "travels" — and are
        // color-graded from query cyan (start) to result amber (end)
        const hops = points.length - 1;
        for (let i = 0; i < hops; i++) {
          const local = clamp(growth * hops - i, 0, 1);
          const hopRgb = lerpRgb(queryRgb, ghostRgb, (i + 1) / hops);
          pushSeg(
            points[i][0], points[i][1], points[i][2],
            points[i + 1][0], points[i + 1][1], points[i + 1][2],
            hopRgb, 0.78, local
          );
        }
        if (state.wpCount > 0) {
          selection.path.waypoints.forEach((wp, wi) => {
            const wpRgb = lerpRgb(queryRgb, ghostRgb, (wi + 1) / hops);
            for (const n of wp.neighbors.slice(0, state.wpCount)) {
              const p = at(n.index);
              pushSeg(wp.pos[0], wp.pos[1], wp.pos[2], p[0], p[1], p[2], wpRgb, 0.2);
            }
          });
        }
      }
    }

    renderer.uploadDynamicLines(new Float32Array(verts));
  }

  // ---------------------------------------------------------------- camera

  function cameraEye() {
    const cp = Math.cos(camera.pitch);
    eye[0] = camera.target[0] + camera.distance * cp * Math.sin(camera.yaw);
    eye[1] = camera.target[1] + camera.distance * Math.sin(camera.pitch);
    eye[2] = camera.target[2] + camera.distance * cp * Math.cos(camera.yaw);
  }

  function updateCamera(dt) {
    let active = false;

    if (camera.tween) {
      const tw = camera.tween;
      tw.t = Math.min(1, tw.t + dt / tw.duration);
      const e = tw.t * tw.t * (3 - 2 * tw.t);
      camera.yaw = lerp(tw.fromYaw, tw.toYaw, e);
      camera.pitch = lerp(tw.fromPitch, tw.toPitch, e);
      camera.distance = lerp(tw.fromDist, tw.toDist, e);
      camera.distanceTarget = camera.distance;
      for (let i = 0; i < 3; i++) {
        camera.target[i] = lerp(tw.fromTarget[i], tw.toTarget[i], e);
      }
      if (tw.t >= 1) camera.tween = null;
      active = true;
    }

    if (state.autoRotate && !pointerState.dragging && !camera.tween) {
      camera.yaw += dt * 0.12;
      active = true;
    }

    if (!pointerState.dragging && !camera.tween) {
      if (Math.abs(camera.velYaw) > 0.0004 || Math.abs(camera.velPitch) > 0.0004) {
        camera.yaw += camera.velYaw * dt;
        camera.pitch = clamp(camera.pitch + camera.velPitch * dt, -1.45, 1.45);
        const decay = Math.exp(-dt * 4.6);
        camera.velYaw *= decay;
        camera.velPitch *= decay;
        active = true;
      } else {
        camera.velYaw = 0;
        camera.velPitch = 0;
      }
    }

    const distDiff = camera.distanceTarget - camera.distance;
    if (Math.abs(distDiff) > 0.001) {
      camera.distance += distDiff * (1 - Math.exp(-dt * 9));
      active = true;
    }

    return active;
  }

  function startCameraTween(toYaw, toPitch, toDist, toTarget, duration = 0.6) {
    camera.velYaw = 0;
    camera.velPitch = 0;
    camera.tween = {
      t: 0,
      duration,
      fromYaw: camera.yaw, toYaw,
      fromPitch: camera.pitch, toPitch,
      fromDist: camera.distance, toDist,
      fromTarget: camera.target.slice(), toTarget: toTarget.slice(),
    };
  }

  function resetCamera() {
    startCameraTween(
      CAMERA_DEFAULTS.yaw,
      CAMERA_DEFAULTS.pitch,
      CAMERA_DEFAULTS.distance,
      CAMERA_DEFAULTS.target
    );
  }

  // ---------------------------------------------------------------- render loop

  function computeMatrices() {
    cameraEye();
    const aspect = Math.max(0.1, state.cssWidth / Math.max(1, state.cssHeight));
    mat4Perspective(projMatrix, Math.PI / 4, aspect, 0.1, 300);
    mat4LookAt(viewMatrix, eye, camera.target, [0, 1, 0]);
    mat4Multiply(viewProj, projMatrix, viewMatrix);
  }

  function computeScreenPositions() {
    if (!screenXY) return;
    const m = viewProj;
    const pos = state.positions;
    const w = state.cssWidth;
    const h = state.cssHeight;
    const limit = visibleCount();
    for (let i = limit; i < state.count; i++) {
      screenXY[i * 3 + 2] = -1;
    }
    for (let i = 0; i < limit; i++) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (cw < 0.02) {
        screenXY[i * 3 + 2] = -1;
        continue;
      }
      const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
      const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
      screenXY[i * 3] = (cx / cw * 0.5 + 0.5) * w;
      screenXY[i * 3 + 1] = (0.5 - cy / cw * 0.5) * h;
      screenXY[i * 3 + 2] = cw;
    }
  }

  let lastFrameTime = performance.now();

  function frame(now) {
    const dt = Math.min(0.05, (now - lastFrameTime) / 1000);
    lastFrameTime = now;

    if (state.prevMix > 0) {
      state.prevMix = Math.max(0, state.prevMix - dt / DIM_DUR);
      state.needsRender = true;
    }

    if (state.animDir !== 0) {
      state.animT += dt * state.animDir * (state.animDir < 0 ? 2 : 1); // clear-out plays 2x
      if (state.animDir > 0 && state.animT >= ANIM_TOTAL) {
        state.animT = ANIM_TOTAL;
        state.animDir = 0;
      } else if (state.animDir < 0 && state.animT <= 0) {
        // fade-out finished: restore the resting cloud styles and reset
        state.animT = ANIM_TOTAL;
        state.animDir = 0;
        if (!state.selection) applySelectionStyles();
      }
      rebuildDynamicLines(lineFactor());
      state.needsRender = true;
    }

    const cameraActive = updateCamera(dt);
    if (cameraActive || state.needsRender) {
      computeMatrices();
      renderer.draw();
      computeScreenPositions();
      state.needsRender = false;
    }
    drawOverlay(now);
    requestAnimationFrame(frame);
  }

  // ---------------------------------------------------------------- overlay (labels, arrows, decorations)

  function pointScreenRadius(index) {
    const w = screenXY[index * 3 + 2];
    if (w <= 0) return 0;
    const size = styleArr ? styleArr[index * 6 + 4] : BASE_POINT_SIZE;
    return clamp(size * state.pointScale * (15 / w) * 0.5, 1, 128);
  }

  function drawOverlay(now) {
    const w = state.cssWidth;
    const h = state.cssHeight;
    octx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    octx.clearRect(0, 0, w, h);
    if (!state.count || !screenXY) return;

    drawAxisHeads();

    const selection = state.selection;
    const lineF = lineFactor();
    const neighF = neighborFactor();

    octx.globalAlpha = lineF;
    if (selection && state.displayVectors) {
      drawVectorArrowHeads(selection);
    }
    if (selection && state.showPath && selection.path && selection.ghost) {
      drawPathDecorations(selection);
    }

    octx.globalAlpha = neighF;
    if (selection && selection.ghost) {
      drawGhostMarker(selection, now);
    }
    octx.globalAlpha = 1;

    drawLabels(selection);

    if (state.pulseIndex >= 0) {
      drawPulseRing(state.pulseIndex, now);
    }

    if (state.hoverMark && selection) {
      drawMarkHover(selection);
    } else if (state.hoverIndex >= 0 && state.hoverIndex !== (selection && selection.queryWord)) {
      drawHover(state.hoverIndex);
    }
  }

  function screenDirection(fromX, fromY, fromZ, toX, toY, toZ) {
    const a = { x: 0, y: 0, w: 0 };
    const b = { x: 0, y: 0, w: 0 };
    if (!projectToScreen(fromX, fromY, fromZ, a) || !projectToScreen(toX, toY, toZ, b)) return null;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy);
    if (len < 0.001) return null;
    return { x: b.x, y: b.y, angle: Math.atan2(dy, dx) };
  }

  function drawArrowHeadAt(x, y, angle, color, size = 8) {
    octx.save();
    octx.translate(x, y);
    octx.rotate(angle);
    octx.beginPath();
    octx.moveTo(0, 0);
    octx.lineTo(-size * 1.35, -size * 0.55);
    octx.lineTo(-size * 1.35, size * 0.55);
    octx.closePath();
    octx.fillStyle = color;
    octx.fill();
    octx.restore();
  }

  function drawAxisHeads() {
    const axes = [
      { to: [GRID_EXTENT, 0, 0], color: PALETTE.axisX, label: 'x' },
      { to: [0, GRID_EXTENT, 0], color: PALETTE.axisY, label: 'y' },
      { to: [0, 0, GRID_EXTENT], color: PALETTE.axisZ, label: 'z' },
    ];
    octx.font = '11px "IBM Plex Mono", monospace';
    for (const axis of axes) {
      const dir = screenDirection(
        axis.to[0] * 0.88, axis.to[1] * 0.88, axis.to[2] * 0.88,
        axis.to[0], axis.to[1], axis.to[2]
      );
      if (!dir) continue;
      drawArrowHeadAt(dir.x, dir.y, dir.angle, hexToCss(axis.color, 0.9), 7);
      octx.fillStyle = hexToCss(axis.color, 0.75);
      octx.fillText(axis.label, dir.x + 8, dir.y - 6);
    }
  }

  function drawVectorArrowHeads(selection) {
    const pos = state.positions;
    const draw = (index, color) => {
      const x = pos[index * 3], y = pos[index * 3 + 1], z = pos[index * 3 + 2];
      const dir = screenDirection(x * 0.9, y * 0.9, z * 0.9, x, y, z);
      if (dir) drawArrowHeadAt(dir.x, dir.y, dir.angle, hexToCss(color, 0.85), 7);
    };
    for (const i of selection.positive) draw(i, PALETTE.query);
    for (const i of selection.negative) draw(i, PALETTE.negative);
    if (selection.ghost) {
      const g = selection.ghost;
      const dir = screenDirection(g[0] * 0.9, g[1] * 0.9, g[2] * 0.9, g[0], g[1], g[2]);
      if (dir) drawArrowHeadAt(dir.x, dir.y, dir.angle, hexToCss(PALETTE.ghost, 0.95), 8);
    }
  }

  function drawDiamond(x, y, size, fill, strokeStyle) {
    octx.save();
    octx.translate(x, y);
    octx.rotate(Math.PI / 4);
    octx.fillStyle = fill;
    octx.strokeStyle = strokeStyle;
    octx.lineWidth = 1.5;
    octx.fillRect(-size, -size, size * 2, size * 2);
    octx.strokeRect(-size, -size, size * 2, size * 2);
    octx.restore();
  }

  function drawGhostMarker(selection, now) {
    const g = selection.ghost;
    if (!projectToScreen(g[0], g[1], g[2], scratchPoint)) return;
    const x = scratchPoint.x;
    const y = scratchPoint.y;
    const pulse = Math.sin(now / 420);

    // bright diamond core with amber halo — nothing else in the scene looks like this
    drawDiamond(x, y, 7, 'rgba(255, 255, 255, 0.97)', hexToCss(PALETTE.ghost, 0.95));

    // crosshair ticks
    octx.strokeStyle = hexToCss(PALETTE.ghost, 0.9);
    octx.lineWidth = 1.6;
    const inner = 10;
    const outer = 16 + pulse * 1.5;
    for (let k = 0; k < 4; k++) {
      const angle = (Math.PI / 2) * k;
      octx.beginPath();
      octx.moveTo(x + Math.cos(angle) * inner, y + Math.sin(angle) * inner);
      octx.lineTo(x + Math.cos(angle) * outer, y + Math.sin(angle) * outer);
      octx.stroke();
    }

    // pulsing ring
    octx.beginPath();
    octx.arc(x, y, 13 + pulse * 3, 0, Math.PI * 2);
    octx.strokeStyle = hexToCss(PALETTE.ghost, 0.4 + pulse * 0.15);
    octx.lineWidth = 1.5;
    octx.stroke();
  }

  function drawPathDecorations(selection) {
    const pos = state.positions;
    const start = selection.path.startIndex;
    const points = [
      [pos[start * 3], pos[start * 3 + 1], pos[start * 3 + 2]],
      ...selection.path.waypoints.map((wp) => wp.pos),
      selection.ghost,
    ];

    const startRgb = hexToRgb(PALETTE.query);
    const endRgb = hexToRgb(PALETTE.ghost);
    const hops = points.length - 1;

    for (let i = 0; i < hops; i++) {
      const a = points[i];
      const b = points[i + 1];
      const dir = screenDirection(
        lerp(a[0], b[0], 0.82), lerp(a[1], b[1], 0.82), lerp(a[2], b[2], 0.82),
        b[0], b[1], b[2]
      );
      const c = lerpRgb(startRgb, endRgb, (i + 1) / hops);
      if (dir) drawArrowHeadAt(dir.x, dir.y, dir.angle, rgbToCss(c[0], c[1], c[2], 0.88), 7.5);
    }

    selection.path.waypoints.forEach((wp, wi) => {
      if (!projectToScreen(wp.pos[0], wp.pos[1], wp.pos[2], scratchPoint)) return;
      const c = lerpRgb(startRgb, endRgb, (wi + 1) / hops);
      drawDiamond(scratchPoint.x, scratchPoint.y, 4.5, rgbToCss(c[0], c[1], c[2], 0.95), 'rgba(6, 12, 21, 0.8)');
    });
  }

  function drawMarkHover(selection) {
    const mark = state.hoverMark;
    let worldPos = null;
    let text = '';
    let color = PALETTE.ghost;

    if (mark.type === 'ghost') {
      worldPos = selection.ghost;
      text = `= ${selection.displayText || selection.text}`;
    } else if (mark.type === 'wp' && selection.path && selection.path.waypoints[mark.index]) {
      const wp = selection.path.waypoints[mark.index];
      worldPos = wp.pos;
      text = `= ${wp.label}`;
      color = PALETTE.waypoint;
    }

    if (!worldPos || !projectToScreen(worldPos[0], worldPos[1], worldPos[2], scratchPoint)) return;
    const x = scratchPoint.x;
    const y = scratchPoint.y;

    octx.beginPath();
    octx.arc(x, y, mark.type === 'ghost' ? 19 : 11, 0, Math.PI * 2);
    octx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    octx.lineWidth = 1.6;
    octx.stroke();

    drawLabelBox(text, x + 22, y, color, []);
  }

  function drawPulseRing(index, now) {
    if (screenXY[index * 3 + 2] <= 0) return;
    const x = screenXY[index * 3];
    const y = screenXY[index * 3 + 1];
    const pulse = Math.sin(now / 260);
    const r = pointScreenRadius(index) + 6 + pulse * 2.5;
    octx.beginPath();
    octx.arc(x, y, r, 0, Math.PI * 2);
    octx.strokeStyle = 'rgba(255, 255, 255, 0.75)';
    octx.lineWidth = 1.8;
    octx.stroke();
  }

  function drawHover(index) {
    if (screenXY[index * 3 + 2] <= 0) return;
    const x = screenXY[index * 3];
    const y = screenXY[index * 3 + 1];
    const r = pointScreenRadius(index) + 4;

    octx.beginPath();
    octx.arc(x, y, r, 0, Math.PI * 2);
    octx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    octx.lineWidth = 1.6;
    octx.stroke();

    let text = state.words[index];
    if (state.selection) {
      const found = state.selection.neighbors.find((n) => n.index === index);
      if (found) text += `  ${found.score.toFixed(3)}`;
    }
    drawLabelBox(text, x + r + 6, y, '#ecf4ff', []);
  }

  const labelRects = [];

  function rectsOverlap(a, b) {
    return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
  }

  function drawLabelBox(text, x, y, color, placed) {
    octx.font = '11.5px "IBM Plex Mono", monospace';
    octx.textBaseline = 'middle';
    const width = octx.measureText(text).width + 14;
    const height = 21;
    const rect = { x, y: y - height / 2, w: width, h: height };

    if (rect.x + rect.w > state.cssWidth - 4) rect.x = x - width - 14;
    rect.x = Math.max(4, rect.x);
    rect.y = clamp(rect.y, 4, state.cssHeight - height - 4);

    for (const other of placed) {
      if (rectsOverlap(rect, other)) return false;
    }
    placed.push(rect);

    octx.beginPath();
    const r = 7;
    octx.moveTo(rect.x + r, rect.y);
    octx.arcTo(rect.x + rect.w, rect.y, rect.x + rect.w, rect.y + rect.h, r);
    octx.arcTo(rect.x + rect.w, rect.y + rect.h, rect.x, rect.y + rect.h, r);
    octx.arcTo(rect.x, rect.y + rect.h, rect.x, rect.y, r);
    octx.arcTo(rect.x, rect.y, rect.x + rect.w, rect.y, r);
    octx.closePath();
    octx.fillStyle = PALETTE.label;
    octx.fill();
    octx.strokeStyle = typeof color === 'string' && color.startsWith('#')
      ? hexToCss(color, 0.4)
      : 'rgba(255, 255, 255, 0.16)';
    octx.lineWidth = 1;
    octx.stroke();

    octx.fillStyle = typeof color === 'string' && color.startsWith('#') ? hexToCss(color, 0.98) : color;
    octx.fillText(text, rect.x + 7, rect.y + rect.h / 2 + 0.5);
    return true;
  }

  function drawLabels(selection) {
    if (!selection || !state.showLabels) return;
    labelRects.length = 0;

    const drawFor = (index, color) => {
      if (screenXY[index * 3 + 2] <= 0) return;
      const x = screenXY[index * 3];
      const y = screenXY[index * 3 + 1];
      if (x < -40 || x > state.cssWidth + 40 || y < -40 || y > state.cssHeight + 40) return;
      drawLabelBox(state.words[index], x + pointScreenRadius(index) + 5, y - 4, color, labelRects);
    };

    octx.globalAlpha = dimFactor();
    for (const i of selection.positive) drawFor(i, PALETTE.query);
    for (const i of selection.negative) drawFor(i, PALETTE.negative);

    octx.globalAlpha = neighborFactor();
    let shown = 0;
    for (const n of selection.neighbors) {
      if (shown >= 24) break;
      drawFor(n.index, PALETTE.neighbor);
      shown++;
    }

    if (state.showPath && selection.path && state.wpCount > 0) {
      const wps = selection.path.waypoints;
      const hops = wps.length + 1;
      const startRgb = hexToRgb(PALETTE.query);
      const endRgb = hexToRgb(PALETTE.ghost);
      wps.forEach((wp, wi) => {
        const c = lerpRgb(startRgb, endRgb, (wi + 1) / hops);
        for (const n of wp.neighbors.slice(0, state.wpCount)) {
          drawFor(n.index, rgbToCss(c[0], c[1], c[2], 0.95));
        }
      });
    }
    octx.globalAlpha = 1;
  }

  // ---------------------------------------------------------------- picking

  function pickMarkerAt(x, y) {
    const selection = state.selection;
    if (!selection) return null;

    if (selection.ghost && projectToScreen(selection.ghost[0], selection.ghost[1], selection.ghost[2], scratchPoint)) {
      if (Math.hypot(scratchPoint.x - x, scratchPoint.y - y) < 15) {
        return { type: 'ghost' };
      }
    }

    if (state.showPath && selection.path) {
      for (let i = 0; i < selection.path.waypoints.length; i++) {
        const wp = selection.path.waypoints[i];
        if (!projectToScreen(wp.pos[0], wp.pos[1], wp.pos[2], scratchPoint)) continue;
        if (Math.hypot(scratchPoint.x - x, scratchPoint.y - y) < 12) {
          return { type: 'wp', index: i };
        }
      }
    }

    return null;
  }

  function centerOnMark(mark) {
    const selection = state.selection;
    if (!selection) return;
    let pos = null;
    if (mark.type === 'ghost') {
      pos = selection.ghost;
    } else if (mark.type === 'wp' && selection.path && selection.path.waypoints[mark.index]) {
      pos = selection.path.waypoints[mark.index].pos;
    }
    if (!pos) return;
    startCameraTween(camera.yaw, camera.pitch, Math.min(camera.distance, 6), pos, 0.55);
  }

  function pickPointAt(x, y) {
    if (!screenXY || !styleArr) return -1;
    let best = -1;
    let bestW = Infinity;
    let bestDist = Infinity;
    const limit = visibleCount();
    for (let i = 0; i < limit; i++) {
      const w = screenXY[i * 3 + 2];
      if (w <= 0) continue;
      if (styleArr[i * 6 + 3] < 0.02) continue; // hidden
      const dx = screenXY[i * 3] - x;
      const dy = screenXY[i * 3 + 1] - y;
      const dist = Math.hypot(dx, dy);
      const radius = Math.max(9, pointScreenRadius(i) + 4);
      if (dist >= radius) continue;
      // depth wins: a big point in front occludes small distant ones behind it
      if (w < bestW - 0.001 || (Math.abs(w - bestW) <= 0.001 && dist < bestDist)) {
        bestW = w;
        bestDist = dist;
        best = i;
      }
    }
    return best;
  }

  // ---------------------------------------------------------------- selection / query flow

  function setStatus(message) {
    dom.statusBar.textContent = message;
  }

  function computeGhostPosition(neighbors) {
    const pos = state.positions;
    const k = Math.min(4, neighbors.length);
    if (!k) return null;
    let wx = 0, wy = 0, wz = 0, wsum = 0;
    for (let i = 0; i < k; i++) {
      const n = neighbors[i];
      const weight = Math.pow(Math.max(n.score, 0) + 0.01, 8);
      wx += pos[n.index * 3] * weight;
      wy += pos[n.index * 3 + 1] * weight;
      wz += pos[n.index * 3 + 2] * weight;
      wsum += weight;
    }
    if (!wsum) return null;
    return [wx / wsum, wy / wsum, wz / wsum];
  }

  async function runQuery(rawText) {
    const text = rawText.trim();
    if (!text) {
      setStatus('Type a word or an expression first.');
      return;
    }
    if (!state.count) {
      setStatus('Data is still loading — one moment.');
      return;
    }

    let ast;
    try {
      ast = parseExpressionSource(text);
    } catch (error) {
      failQuery(error.message);
      return;
    }

    const topN = state.topN;
    const token = ++state.queryToken;

    try {
      if (ast.type === 'word') {
        const index = lookupWord(ast.value);
        if (index === undefined) {
          failQuery(unknownWordMessage(ast.value));
          return;
        }
        const neighbors = await searchNeighbors(getVector(index), topN, new Set([index]));
        if (token !== state.queryToken) return;
        applySelection({
          kind: 'word',
          text,
          queryWord: index,
          positive: [index],
          negative: [],
          neighbors,
          ghost: null,
          path: null,
        });
      } else {
        const evaluated = evaluateExpressionNode(ast);
        if (evaluated.kind !== 'vector') {
          throw buildExpressionError('The expression must include at least one word');
        }
        const terms = collectSignedWords(ast);
        const positive = [...new Set(terms.positive.map((w) => state.wordIndex.get(w)).filter((v) => v !== undefined))];
        const negative = [...new Set(terms.negative.map((w) => state.wordIndex.get(w)).filter((v) => v !== undefined))]
          .filter((i) => !positive.includes(i));
        const excludeSet = new Set([...positive, ...negative]);
        const neighbors = await searchNeighbors(evaluated.value, topN, excludeSet);
        if (token !== state.queryToken) return;
        const jump = await buildJump(ast, excludeSet);
        if (token !== state.queryToken) return;
        applySelection({
          kind: 'expression',
          text,
          displayText: text,
          queryWord: null,
          positive,
          negative,
          neighbors,
          ghost: computeGhostPosition(neighbors),
          jump,
          activeStep: jump ? jump.stepVectors.length - 1 : -1,
          path: jump
            ? { startIndex: jump.startIndex, waypoints: jump.waypoints.filter(Boolean) }
            : null,
        });
      }
      state.lastQueryText = text;
    } catch (error) {
      failQuery(error.message);
    }
  }

  // Build the arithmetic "throwaround" data: cumulative vector + label per
  // step (one step per term, so `man + 2*woman` is a single hop). Intermediate
  // stops have no true position in the projection, so each is placed at the
  // similarity-weighted centroid of its nearest words. waypoints[s-1] belongs
  // to step s and is null when that partial sum cancels to zero.
  async function buildJump(ast, excludeSet) {
    const terms = linearTerms(ast);
    if (!terms || terms.length < 2) return null;
    const indices = terms.map((term) => lookupWord(term.word));
    if (indices.some((index) => index === undefined)) return null;

    const stepVectors = [];
    const stepLabels = [];
    const cum = new Float32Array(state.dim);
    let labelText = '';
    for (let s = 0; s < terms.length; s++) {
      const vector = getVector(indices[s]);
      for (let j = 0; j < state.dim; j++) cum[j] += vector[j] * terms[s].weight;
      labelText += formatTerm(terms[s], s === 0);
      stepVectors.push(new Float32Array(cum));
      stepLabels.push(labelText);
    }

    const waypoints = [];
    for (let s = 1; s < terms.length - 1; s++) {
      let entry = null;
      try {
        const wpNeighbors = await searchNeighbors(stepVectors[s], 4, excludeSet);
        const pos = computeGhostPosition(wpNeighbors);
        if (pos) {
          entry = { pos, label: stepLabels[s], neighbors: wpNeighbors.slice(0, 3) };
        }
      } catch (error) {
        // partial sum cancels to a zero vector — no waypoint for this stop
      }
      waypoints.push(entry);
    }

    return { terms, indices, stepVectors, stepLabels, waypoints, startIndex: indices[0] };
  }

  function failQuery(message) {
    dom.resultTitle.textContent = 'Invalid query';
    dom.resultSubtitle.textContent = message;
    setStatus(message);
  }

  // Jump to a step of the expression: treat the partial sum through terms
  // 0..stepIndex as the final destination — full neighbor search, truncated
  // path, results panel and all.
  async function applyStep(stepIndex) {
    const selection = state.selection;
    if (!selection || !selection.jump) return;
    const jump = selection.jump;
    stepIndex = clamp(stepIndex, 0, jump.stepVectors.length - 1);
    if (stepIndex === selection.activeStep) return;
    selection.activeStep = stepIndex;
    updateJumper(selection);

    let neighbors = [];
    try {
      const excludeSet = new Set(jump.indices.slice(0, stepIndex + 1));
      neighbors = await searchNeighbors(jump.stepVectors[stepIndex], state.topN, excludeSet);
    } catch (error) {
      // zero partial sum — show the stop with no neighbors
    }
    if (state.selection !== selection || selection.activeStep !== stepIndex) return;

    const positive = [];
    const negative = [];
    for (let s = 0; s <= stepIndex; s++) {
      const index = jump.indices[s];
      const list = jump.terms[s].weight >= 0 ? positive : negative;
      if (!list.includes(index)) list.push(index);
    }
    selection.positive = positive;
    selection.negative = negative.filter((index) => !positive.includes(index));
    selection.neighbors = neighbors;
    selection.displayText = jump.stepLabels[stepIndex];

    if (stepIndex === 0) {
      selection.queryWord = jump.startIndex;
      selection.ghost = null;
      selection.path = null;
    } else {
      selection.queryWord = null;
      selection.ghost = computeGhostPosition(neighbors);
      selection.path = {
        startIndex: jump.startIndex,
        waypoints: jump.waypoints.slice(0, stepIndex - 1).filter(Boolean),
      };
    }

    state.hoverMark = null;
    if (state.animEnabled) {
      renderer.snapshotPrev();
      state.prevMix = 1;
    }
    applySelectionStyles();
    // animate the jump like a re-query: background stays dimmed, lines
    // travel to the new stop, neighbors bloom in
    state.animT = state.animEnabled ? LINE_START : ANIM_TOTAL;
    state.animDir = state.animT < ANIM_TOTAL ? 1 : 0;
    rebuildDynamicLines(lineFactor());
    updateResultsPanel();
    state.needsRender = true;

    const top = neighbors[0];
    setStatus(top
      ? `Stop ${stepIndex + 1}/${jump.stepLabels.length} — closest to ${selection.displayText}: ${state.words[top.index]} (${top.score.toFixed(3)})`
      : `Stop ${stepIndex + 1}/${jump.stepLabels.length} — no neighbors here.`);
  }

  function updateJumper(selection) {
    const jump = selection && selection.jump;
    if (!jump) {
      dom.jumper.hidden = true;
      return;
    }
    dom.jumper.hidden = false;
    dom.jumpSlider.max = String(jump.stepLabels.length - 1);
    dom.jumpSlider.value = String(selection.activeStep);
    dom.jumpChips.innerHTML = '';
    jump.stepLabels.forEach((label, index) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip-btn jump-chip' + (index === selection.activeStep ? ' active' : '');
      chip.textContent = label;
      chip.addEventListener('click', () => applyStep(index));
      dom.jumpChips.appendChild(chip);
    });
  }

  function applySelection(selection) {
    const hadSelection = !!state.selection;
    state.selection = selection;
    state.pulseIndex = -1;
    state.hoverMark = null;
    if (hadSelection && state.animEnabled) {
      // outgoing selection cross-fades away while the new one animates in
      renderer.snapshotPrev();
      state.prevMix = 1;
    }
    applySelectionStyles();
    // re-queries skip the dim phase: the background is already dark, so no
    // un-dim/re-dim flash — only the lines + neighbors replay
    state.animT = state.animEnabled ? (hadSelection ? LINE_START : 0) : ANIM_TOTAL;
    state.animDir = state.animT < ANIM_TOTAL ? 1 : 0;
    rebuildDynamicLines(lineFactor());
    updateResultsPanel();
    updateJumper(selection);
    state.needsRender = true;

    const top = selection.neighbors[0];
    if (top) {
      setStatus(`Closest match: ${state.words[top.index]} (${top.score.toFixed(3)})`);
    } else {
      setStatus('No neighbors found.');
    }
  }

  function clearSelection() {
    const hadSelection = !!state.selection;
    state.selection = null;
    state.pulseIndex = -1;
    state.hoverMark = null;
    state.lastQueryText = '';
    dom.jumper.hidden = true;
    if (hadSelection && state.animEnabled) {
      // keep the old styles and play the animation backwards; frame()
      // restores the resting styles once it reaches zero
      state.animDir = -1;
    } else {
      applySelectionStyles();
      rebuildDynamicLines();
      state.animT = ANIM_TOTAL;
      state.animDir = 0;
    }
    updateResultsPanel();
    state.needsRender = true;
    setStatus(`${state.count.toLocaleString()} words loaded. Click a point or type a query.`);
  }

  function updateResultsPanel() {
    const selection = state.selection;
    dom.resultList.innerHTML = '';
    dom.queryChips.innerHTML = '';

    if (!selection) {
      dom.resultTitle.textContent = 'Nothing selected';
      dom.resultSubtitle.textContent = 'Search a word, write vector math, or click any point in the cloud.';
      return;
    }

    if (selection.kind === 'word') {
      dom.resultTitle.textContent = `"${state.words[selection.queryWord]}"`;
      dom.resultSubtitle.textContent = `${selection.neighbors.length} nearest neighbors by cosine similarity.`;
    } else {
      const jump = selection.jump;
      const atStop = jump && selection.activeStep < jump.stepLabels.length - 1;
      dom.resultTitle.textContent = selection.displayText || selection.text;
      dom.resultSubtitle.textContent = atStop
        ? `Stop ${selection.activeStep + 1} of ${jump.stepLabels.length} — treating this partial sum as the destination.`
        : `${selection.neighbors.length} nearest to the computed vector — input words excluded.`;
    }

    const addChip = (index, sign) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = `word-chip ${sign >= 0 ? 'positive' : 'negative'}`;
      chip.textContent = `${sign >= 0 ? '+' : '−'} ${state.words[index]}`;
      chip.addEventListener('click', () => focusWord(state.words[index]));
      dom.queryChips.appendChild(chip);
    };
    if (selection.kind === 'expression') {
      selection.positive.forEach((i) => addChip(i, 1));
      selection.negative.forEach((i) => addChip(i, -1));
    }

    selection.neighbors.forEach((neighbor, index) => {
      const li = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'result-row';

      const rank = document.createElement('span');
      rank.className = 'rank';
      rank.textContent = String(index + 1);

      const word = document.createElement('span');
      word.className = 'word';
      word.textContent = state.words[neighbor.index];

      const bar = document.createElement('span');
      bar.className = 'bar';
      const fill = document.createElement('i');
      fill.style.width = `${clamp(Math.max(neighbor.score, 0) * 100, 4, 100)}%`;
      bar.appendChild(fill);

      const score = document.createElement('span');
      score.className = 'score';
      score.textContent = neighbor.score.toFixed(3);

      button.append(rank, word, bar, score);
      button.addEventListener('click', () => focusWord(state.words[neighbor.index]));
      button.addEventListener('mouseenter', () => { state.pulseIndex = neighbor.index; });
      button.addEventListener('mouseleave', () => {
        if (state.pulseIndex === neighbor.index) state.pulseIndex = -1;
      });
      li.appendChild(button);
      dom.resultList.appendChild(li);
    });
  }

  function focusWord(word) {
    // tokens that don't parse as a plain identifier (punctuation, digits,
    // hyphens the vocab-greedy pass can't resolve) get quoted
    const plain = /^[\p{L}_][\p{L}\p{N}_']*$/u.test(word)
      || (/^[\p{L}_]/u.test(word) && state.wordIndex.has(word)); // greedy pass resolves these
    const delim = ['"', "'", '`'].find((d) => !word.includes(d));
    const query = plain || !delim ? word : `${delim}${word}${delim}`;
    dom.queryInput.value = query;
    hideSuggestions();
    runQuery(query);
  }

  // ---------------------------------------------------------------- autocomplete

  const suggest = {
    items: [],
    active: -1,
    token: null,
  };

  function currentToken() {
    const el = dom.queryInput;
    const pos = el.selectionStart ?? el.value.length;
    const text = el.value;
    let start = pos;
    while (start > 0 && IDENT_PART.test(text[start - 1])) start--;
    let end = pos;
    while (end < text.length && IDENT_PART.test(text[end])) end++;
    return { start, end, text: text.slice(start, pos).toLowerCase() };
  }

  function updateSuggestions() {
    const token = currentToken();
    if (!token.text || token.text.length < 1 || !state.count) {
      hideSuggestions();
      return;
    }

    const prefix = [];
    const substr = [];
    const limit = visibleCount();
    for (let i = 0; i < limit && prefix.length < 8; i++) {
      const word = state.words[i];
      if (word === token.text) continue;
      if (word.startsWith(token.text)) prefix.push(word);
    }
    if (prefix.length < 8) {
      for (let i = 0; i < limit && prefix.length + substr.length < 8; i++) {
        const word = state.words[i];
        if (word === token.text || word.startsWith(token.text)) continue;
        if (word.includes(token.text)) substr.push(word);
      }
    }

    suggest.items = [...prefix, ...substr];
    suggest.active = -1;
    suggest.token = token;

    if (!suggest.items.length) {
      hideSuggestions();
      return;
    }

    dom.suggestions.innerHTML = '';
    suggest.items.forEach((word, index) => {
      const li = document.createElement('li');
      li.textContent = word;
      li.addEventListener('mousedown', (event) => {
        event.preventDefault();
        applySuggestion(index, true);
      });
      dom.suggestions.appendChild(li);
    });
    dom.suggestions.hidden = false;
  }

  function hideSuggestions() {
    dom.suggestions.hidden = true;
    suggest.items = [];
    suggest.active = -1;
  }

  function highlightSuggestion() {
    [...dom.suggestions.children].forEach((li, index) => {
      li.classList.toggle('active', index === suggest.active);
    });
  }

  function applySuggestion(index, run) {
    const word = suggest.items[index];
    if (!word || !suggest.token) return;
    const el = dom.queryInput;
    const value = el.value;
    const before = value.slice(0, suggest.token.start);
    const after = value.slice(suggest.token.end);
    el.value = before + word + after;
    const caret = before.length + word.length;
    el.setSelectionRange(caret, caret);
    hideSuggestions();
    if (run) runQuery(el.value);
  }

  // ---------------------------------------------------------------- events

  function resize() {
    const rect = dom.stage.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    if (rect.width === state.cssWidth && rect.height === state.cssHeight && dpr === state.dpr) return;
    state.cssWidth = rect.width;
    state.cssHeight = rect.height;
    state.dpr = dpr;
    const pw = Math.max(1, Math.round(rect.width * dpr));
    const ph = Math.max(1, Math.round(rect.height * dpr));
    dom.glCanvas.width = pw;
    dom.glCanvas.height = ph;
    dom.overlay.width = pw;
    dom.overlay.height = ph;
    state.needsRender = true;
  }

  function bindStageEvents() {
    const stage = dom.stage;

    stage.addEventListener('contextmenu', (event) => event.preventDefault());

    const overCanvas = (event) =>
      event.target === dom.glCanvas || event.target === dom.stage || event.target === dom.overlay;

    stage.addEventListener('pointerdown', (event) => {
      if (!overCanvas(event)) return; // e.g. the jumper controls
      stage.setPointerCapture(event.pointerId);
      pointerState.pointers.set(event.pointerId, { x: event.offsetX, y: event.offsetY });

      if (pointerState.pointers.size === 2) {
        const pts = [...pointerState.pointers.values()];
        pointerState.pinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        pointerState.mode = 'pinch';
        pointerState.dragging = true;
        return;
      }

      camera.tween = null;
      camera.velYaw = 0;
      camera.velPitch = 0;
      pointerState.dragging = true;
      pointerState.mode = (event.button === 2 || event.button === 1 || event.ctrlKey || event.shiftKey) ? 'pan' : 'orbit';
      pointerState.lastX = event.offsetX;
      pointerState.lastY = event.offsetY;
      pointerState.downX = event.offsetX;
      pointerState.downY = event.offsetY;
      pointerState.moved = 0;
      pointerState.lastMoveTime = performance.now();
      stage.classList.add('grabbing');
    });

    stage.addEventListener('pointermove', (event) => {
      const tracked = pointerState.pointers.get(event.pointerId);
      if (tracked) {
        tracked.x = event.offsetX;
        tracked.y = event.offsetY;
      }

      if (pointerState.mode === 'pinch' && pointerState.pointers.size === 2) {
        const pts = [...pointerState.pointers.values()];
        const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        if (pointerState.pinchDist > 0 && dist > 0) {
          camera.distanceTarget = clamp(camera.distanceTarget * (pointerState.pinchDist / dist), 0.2, 120);
        }
        pointerState.pinchDist = dist;
        return;
      }

      if (!pointerState.dragging) {
        if (!overCanvas(event)) {
          state.hoverIndex = -1;
          state.hoverMark = null;
          stage.classList.remove('point-hover');
          return;
        }
        const mark = pickMarkerAt(event.offsetX, event.offsetY);
        state.hoverMark = mark;
        const picked = mark ? -1 : pickPointAt(event.offsetX, event.offsetY);
        if (picked !== state.hoverIndex) {
          state.hoverIndex = picked;
        }
        stage.classList.toggle('point-hover', picked >= 0 || !!mark);
        return;
      }

      const now = performance.now();
      const dt = Math.max(1, now - pointerState.lastMoveTime) / 1000;
      const dx = event.offsetX - pointerState.lastX;
      const dy = event.offsetY - pointerState.lastY;
      pointerState.lastX = event.offsetX;
      pointerState.lastY = event.offsetY;
      pointerState.lastMoveTime = now;
      pointerState.moved += Math.abs(dx) + Math.abs(dy);

      if (pointerState.mode === 'pan') {
        const panScale = camera.distance * 0.0014;
        const sinYaw = Math.sin(camera.yaw);
        const cosYaw = Math.cos(camera.yaw);
        // camera right vector on ground plane + view-up approximation
        camera.target[0] -= (cosYaw * dx) * panScale;
        camera.target[2] += (sinYaw * dx) * panScale;
        camera.target[1] += dy * panScale;
      } else {
        const dYaw = -dx * 0.0052;
        const dPitch = dy * 0.0052;
        camera.yaw += dYaw;
        camera.pitch = clamp(camera.pitch + dPitch, -1.45, 1.45);
        // clamp hard: tiny dt between pointer events can spike dYaw/dt and
        // make the camera "snap" after release
        camera.velYaw = clamp(lerp(camera.velYaw, dYaw / dt, 0.3), -1.4, 1.4);
        camera.velPitch = clamp(lerp(camera.velPitch, dPitch / dt, 0.3), -1.4, 1.4);
      }
      state.needsRender = true;
    });

    const endPointer = (event) => {
      pointerState.pointers.delete(event.pointerId);
      try {
        stage.releasePointerCapture(event.pointerId);
      } catch (error) { /* already released */ }

      if (pointerState.mode === 'pinch') {
        if (pointerState.pointers.size < 2) {
          pointerState.dragging = false;
          pointerState.mode = 'orbit';
        }
        return;
      }

      if (!pointerState.dragging) return;
      pointerState.dragging = false;
      stage.classList.remove('grabbing');

      // inertia only for a genuine flick — a pause before release means stop
      if (performance.now() - pointerState.lastMoveTime > 90) {
        camera.velYaw = 0;
        camera.velPitch = 0;
      }

      // treat as click when barely moved
      if (pointerState.moved < 5 && pointerState.mode === 'orbit' && event.type === 'pointerup') {
        camera.velYaw = 0;
        camera.velPitch = 0;
        // equation stop markers: click is inert (double-click centers)
        if (pickMarkerAt(event.offsetX, event.offsetY)) return;
        const picked = pickPointAt(event.offsetX, event.offsetY);
        if (picked >= 0) {
          focusWord(state.words[picked]);
        } else if (state.selection) {
          clearSelection();
        }
      }
    };

    stage.addEventListener('pointerup', endPointer);
    stage.addEventListener('pointercancel', endPointer);
    stage.addEventListener('pointerleave', () => {
      if (!pointerState.dragging) {
        state.hoverIndex = -1;
        state.hoverMark = null;
        stage.classList.remove('point-hover');
      }
    });

    stage.addEventListener('wheel', (event) => {
      if (!overCanvas(event)) return;
      event.preventDefault();
      camera.tween = null;
      camera.distanceTarget = clamp(camera.distanceTarget * Math.exp(event.deltaY * 0.0011), 0.2, 120);
    }, { passive: false });

    stage.addEventListener('dblclick', (event) => {
      const mark = pickMarkerAt(event.offsetX, event.offsetY);
      if (mark) {
        centerOnMark(mark);
        return;
      }
      const picked = pickPointAt(event.offsetX, event.offsetY);
      if (picked < 0) return;
      const pos = state.positions;
      startCameraTween(
        camera.yaw,
        camera.pitch,
        Math.min(camera.distance, 8),
        [pos[picked * 3], pos[picked * 3 + 1], pos[picked * 3 + 2]],
        0.55
      );
    });
  }

  function bindUiEvents() {
    dom.runButton.addEventListener('click', () => runQuery(dom.queryInput.value));
    dom.clearButton.addEventListener('click', () => {
      dom.queryInput.value = '';
      clearSelection();
    });
    dom.resetButton.addEventListener('click', resetCamera);

    dom.queryInput.addEventListener('input', updateSuggestions);
    dom.queryInput.addEventListener('blur', () => setTimeout(hideSuggestions, 120));
    dom.queryInput.addEventListener('keydown', (event) => {
      const open = !dom.suggestions.hidden && suggest.items.length > 0;
      if (event.key === 'ArrowDown' && open) {
        event.preventDefault();
        suggest.active = (suggest.active + 1) % suggest.items.length;
        highlightSuggestion();
      } else if (event.key === 'ArrowUp' && open) {
        event.preventDefault();
        suggest.active = (suggest.active - 1 + suggest.items.length) % suggest.items.length;
        highlightSuggestion();
      } else if (event.key === 'Tab' && open) {
        event.preventDefault();
        applySuggestion(suggest.active >= 0 ? suggest.active : 0, false);
      } else if (event.key === 'Enter') {
        event.preventDefault();
        if (open && suggest.active >= 0) {
          applySuggestion(suggest.active, true);
        } else {
          hideSuggestions();
          runQuery(dom.queryInput.value);
        }
      } else if (event.key === 'Escape') {
        if (open) {
          hideSuggestions();
        } else {
          dom.queryInput.blur();
        }
      }
    });

    dom.exampleChips.addEventListener('click', (event) => {
      const button = event.target.closest('[data-q]');
      if (!button) return;
      dom.queryInput.value = button.dataset.q;
      runQuery(button.dataset.q);
    });

    let topNTimer = 0;
    dom.topNInput.addEventListener('input', () => {
      state.topN = Number(dom.topNInput.value) || 10;
      dom.topNValue.textContent = String(state.topN);
      clearTimeout(topNTimer);
      topNTimer = setTimeout(() => {
        if (state.lastQueryText) runQuery(state.lastQueryText);
      }, 180);
    });

    dom.sizeInput.addEventListener('input', () => {
      state.pointScale = Number(dom.sizeInput.value) / 100;
      dom.sizeValue.textContent = `${state.pointScale.toFixed(1)}×`;
      state.needsRender = true;
    });

    dom.fadeInput.addEventListener('input', () => {
      state.depthFade = Number(dom.fadeInput.value) / 100;
      dom.fadeValue.textContent = `${dom.fadeInput.value}%`;
      state.needsRender = true;
    });

    let limitTimer = 0;
    const setRenderLimit = (raw) => {
      if (!state.count) return;
      const limit = clamp(Math.round(Number(raw) || state.count), Math.min(100, state.count), state.count);
      if (limit === state.renderLimit) return;
      state.renderLimit = limit;
      dom.limitInput.value = String(limit);
      dom.limitNumber.value = String(limit);
      state.needsRender = true;
      clearTimeout(limitTimer);
      limitTimer = setTimeout(() => {
        if (state.lastQueryText) {
          runQuery(state.lastQueryText);
        } else {
          setStatus(`Showing the ${limit.toLocaleString()} most frequent words.`);
        }
      }, 220);
    };
    dom.limitInput.addEventListener('input', () => setRenderLimit(dom.limitInput.value));
    dom.limitNumber.addEventListener('change', () => setRenderLimit(dom.limitNumber.value));

    dom.jumpSlider.addEventListener('input', () => {
      applyStep(Number(dom.jumpSlider.value));
    });

    dom.datasetSelect.addEventListener('change', () => {
      loadSizedDataset(Number(dom.datasetSelect.value));
    });

    dom.dimInput.addEventListener('input', () => {
      state.dimming = Number(dom.dimInput.value) / 100;
      dom.dimValue.textContent = `${dom.dimInput.value}%`;
      applySelectionStyles();
      state.needsRender = true;
    });

    dom.displayMode.addEventListener('click', (event) => {
      const button = event.target.closest('[data-mode]');
      if (!button) return;
      state.displayVectors = button.dataset.mode === 'vectors';
      [...dom.displayMode.children].forEach((child) => {
        child.classList.toggle('active', child === button);
      });
      rebuildDynamicLines();
      state.needsRender = true;
    });

    dom.pathToggle.addEventListener('change', () => {
      state.showPath = dom.pathToggle.checked;
      applySelectionStyles();
      rebuildDynamicLines();
      state.needsRender = true;
    });

    dom.wpInput.addEventListener('input', () => {
      state.wpCount = Number(dom.wpInput.value) || 0;
      dom.wpValue.textContent = String(state.wpCount);
      applySelectionStyles();
      rebuildDynamicLines();
      state.needsRender = true;
    });

    dom.linesToggle.addEventListener('change', () => {
      state.showLines = dom.linesToggle.checked;
      rebuildDynamicLines();
      state.needsRender = true;
    });

    dom.labelsToggle.addEventListener('change', () => {
      state.showLabels = dom.labelsToggle.checked;
    });

    dom.rotateToggle.addEventListener('change', () => {
      state.autoRotate = dom.rotateToggle.checked;
    });

    dom.animToggle.addEventListener('change', () => {
      state.animEnabled = dom.animToggle.checked;
      if (!state.animEnabled && state.animDir > 0) {
        state.animT = ANIM_TOTAL;
        state.animDir = 0;
        rebuildDynamicLines(1);
        state.needsRender = true;
      }
    });

    window.addEventListener('keydown', (event) => {
      const typing = document.activeElement && ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName);
      if (event.key === '/' && !typing) {
        event.preventDefault();
        dom.queryInput.focus();
        dom.queryInput.select();
      } else if (event.key === 'Escape' && !typing && state.selection) {
        clearSelection();
      }
    });

    window.addEventListener('resize', resize);
    if (typeof ResizeObserver !== 'undefined') {
      new ResizeObserver(resize).observe(dom.stage);
    }
  }

  // ---------------------------------------------------------------- github stars

  function formatCount(n) {
    if (n < 1000) return String(n);
    if (n < 999500) {
      const v = n / 1000;
      return `${(v >= 100 ? Math.round(v) : Math.round(v * 10) / 10).toString().replace(/\.0$/, '')}K`;
    }
    return `${(Math.round((n / 1e6) * 10) / 10).toString().replace(/\.0$/, '')}M`;
  }

  async function fetchStarCount() {
    try {
      const response = await fetch('https://api.github.com/repos/Allexsen/embedding3d');
      if (!response.ok) return;
      const data = await response.json();
      if (typeof data.stargazers_count !== 'number') return;
      dom.starCount.textContent = `★ ${formatCount(data.stargazers_count)}`;
      dom.starCount.hidden = false;
    } catch (error) {
      // offline or rate-limited — the link works fine without the count
    }
  }

  // ---------------------------------------------------------------- boot

  function start() {
    renderer.init();
    resize();
    bindStageEvents();
    bindUiEvents();

    state.topN = Number(dom.topNInput.value) || 10;
    dom.topNValue.textContent = String(state.topN);
    state.pointScale = Number(dom.sizeInput.value) / 100;
    state.showPath = dom.pathToggle.checked;
    state.wpCount = Number(dom.wpInput.value) || 0;
    dom.wpValue.textContent = String(state.wpCount);

    loadData().then(() => {
      clearSelection();
    });
    fetchStarCount();

    requestAnimationFrame((now) => {
      lastFrameTime = now;
      requestAnimationFrame(frame);
    });
  }

  start();
}());

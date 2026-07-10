import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';

// transformers.js: allow remote model download, cache in browser
env.allowLocalModels = false;

(function () {
  'use strict';

  const MAX_DPR = 2;
  const GRID_EXTENT = 6;
  const BASE_POINT_SIZE = 2.7;
  const CAMERA_DEFAULTS = { yaw: -0.65, pitch: 0.3, distance: 15, target: [0, 0.4, 0] };
  const SLOT_COLORS = ['#78e0ff', '#ffc66d', '#ff8db3', '#8de28f', '#c9a4ff', '#ff9d7e'];

  const EXAMPLES = {
    gpt: [
      'is chatgpt actually thinking or just autocomplete on steroids',
      'Do large language models actually understand meaning or are they just predicting tokens?',
    ],
    results: [
      'yeah it worked way better than the old way lol',
      'method achieved statistically significant improvements',
    ],
    quantum: [
      'can you use quantum stuff to text faster than light or nah?',
      'Does quantum entanglement actually enable faster-than-light communication?',
    ],
    dopamine: [
      'dopamine is basically the feel-good hit you get when something goes better than you expected',
      'dopamine mediates reward prediction error signaling in mesolimbic pathways',
    ],
  };

  const dom = {
    stage: document.getElementById('stage'),
    glCanvas: document.getElementById('glCanvas'),
    overlay: document.getElementById('overlayCanvas'),
    modelSelect: document.getElementById('modelSelect'),
    datasetSelect: document.getElementById('datasetSelect'),
    precisionSelect: document.getElementById('precisionSelect'),
    starCount: document.getElementById('starCount'),
    slots: document.getElementById('slots'),
    addSlot: document.getElementById('addSlot'),
    runButton: document.getElementById('runButton'),
    resetButton: document.getElementById('resetButton'),
    clearButton: document.getElementById('clearButton'),
    topNInput: document.getElementById('topNInput'),
    topNValue: document.getElementById('topNValue'),
    sizeInput: document.getElementById('sizeInput'),
    sizeValue: document.getElementById('sizeValue'),
    fadeInput: document.getElementById('fadeInput'),
    fadeValue: document.getElementById('fadeValue'),
    dimInput: document.getElementById('dimInput'),
    dimValue: document.getElementById('dimValue'),
    colorModeToggle: document.getElementById('colorModeToggle'),
    legend: document.getElementById('legend'),
    morph: document.getElementById('morph'),
    morphSlider: document.getElementById('morphSlider'),
    morphLabel: document.getElementById('morphLabel'),
    followToggle: document.getElementById('followToggle'),
    precomputeToggle: document.getElementById('precomputeToggle'),
    playButton: document.getElementById('playButton'),
    metrics: document.getElementById('metrics'),
    resultList: document.getElementById('resultList'),
    resultSummary: document.getElementById('resultSummary'),
    loadingPanel: document.getElementById('loadingPanel'),
    loadingItems: document.getElementById('loadingItems'),
    statusBar: document.getElementById('statusBar'),
  };

  const octx = dom.overlay.getContext('2d');

  const state = {
    count: 0,
    dim: 384,
    positions: null,       // Float32Array count*3
    sourceIdx: null,       // Uint8Array count
    sources: [],           // [{name, color}]
    baseColors: null,      // Float32Array count*3
    textBlob: null,        // Uint8Array
    textOffsets: null,     // Uint32Array
    textDecoder: new TextDecoder(),

    topN: 15,
    pointScale: 0.4,
    depthFade: 0.8,
    dimming: 0.4,
    sourceColorNodes: true, // highlighted nodes keep their source hue
    followResult: false,    // camera glides to the morph point while scrubbing
    precise: false,         // precompute all morph steps for lag-free scrubbing
    playing: false,         // autoplay running

    slots: [],             // user input phrasings: [{text, color, vec, neighbors, ghost}]
    display: [],           // what is actually rendered: [{color, neighbors, ghost, text}]
                           //   normally mirrors the query slots; during morph it is
                           //   a single interpolated group — kept separate so morph
                           //   never mutates the user's phrasing slots
    hoverIndex: -1,
    model: 'mpnet',        // active embedding model key
    browserId: 'Xenova/all-mpnet-base-v2',
    precision: 'int8',
    tierSize: 40000,

    cssWidth: 0, cssHeight: 0, dpr: 1,
    cloudRadius: 4.4,
    needsRender: true,
  };

  const camera = {
    yaw: CAMERA_DEFAULTS.yaw, pitch: CAMERA_DEFAULTS.pitch,
    distance: CAMERA_DEFAULTS.distance, distanceTarget: CAMERA_DEFAULTS.distance,
    target: CAMERA_DEFAULTS.target.slice(),
    velYaw: 0, velPitch: 0, tween: null,
    follow: null, // [x,y,z] live target the camera springs toward while scrubbing
  };

  const pointerState = {
    dragging: false, mode: 'orbit', lastX: 0, lastY: 0,
    downX: 0, downY: 0, moved: 0, lastMoveTime: 0,
    pointers: new Map(), pinchDist: 0,
  };

  const projMatrix = new Float32Array(16);
  const viewMatrix = new Float32Array(16);
  const viewProj = new Float32Array(16);
  const eye = [0, 0, 1];
  let screenXY = null;

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const lerp = (a, b, t) => a + (b - a) * t;

  function hexToRgb(hex) {
    return [parseInt(hex.slice(1, 3), 16) / 255, parseInt(hex.slice(3, 5), 16) / 255, parseInt(hex.slice(5, 7), 16) / 255];
  }
  function hexToCss(hex, a) {
    const [r, g, b] = hexToRgb(hex);
    return `rgba(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)}, ${a})`;
  }

  // ---------------------------------------------------------------- mat4

  function mat4Perspective(out, fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    out.fill(0);
    out[0] = f / aspect; out[5] = f;
    out[10] = (far + near) / (near - far); out[11] = -1;
    out[14] = (2 * far * near) / (near - far);
    return out;
  }
  function mat4LookAt(out, eyeV, center, up) {
    let zx = eyeV[0] - center[0], zy = eyeV[1] - center[1], zz = eyeV[2] - center[2];
    let len = Math.hypot(zx, zy, zz) || 1; zx /= len; zy /= len; zz /= len;
    let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
    len = Math.hypot(xx, xy, xz) || 1; xx /= len; xy /= len; xz /= len;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
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
    if (w < 0.02) { out.w = -1; return false; }
    const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
    const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
    out.x = (cx / w * 0.5 + 0.5) * state.cssWidth;
    out.y = (0.5 - cy / w * 0.5) * state.cssHeight;
    out.w = w;
    return true;
  }

  // ---------------------------------------------------------------- text access

  function passageText(index) {
    const start = state.textOffsets[index];
    const end = state.textOffsets[index + 1];
    return state.textDecoder.decode(state.textBlob.subarray(start, end));
  }

  // ---------------------------------------------------------------- WebGL (from word app, source-fog + fade)

  const POINT_VS = `
    attribute vec3 aPos;
    attribute vec4 aColor;
    attribute vec2 aExtra;
    attribute vec3 aBase;
    uniform mat4 uViewProj;
    uniform vec3 uEye;
    uniform float uDist;
    uniform float uSizeScale;
    uniform float uDim;         // 0 = flat resting cloud, 1 = background dimmed
    uniform float uHi;          // 0..1 neighbor highlight-in (color/size/alpha)
    uniform float uBaseSize;
    uniform float uFade;
    uniform float uRadius;
    uniform float uSourceColor; // 1 = highlighted nodes keep their source hue
    uniform float uDimAlpha;    // dimmed alpha for non-neighbors during selection
    varying vec4 vColor;
    varying float vRing;
    varying float vSize;
    varying float vFog;
    void main() {
      vec4 clip = uViewProj * vec4(aPos, 1.0);
      gl_Position = clip;
      float w = max(clip.w, 0.1);

      // Two independent blends so a swap can re-animate neighbors (uHi) while
      // holding the background dimmed (uDim):
      //   background points: alpha 0.52 -> uDimAlpha by uDim (color unchanged)
      //   neighbor points:   fade in color/size/alpha by uHi
      float neighbor = step(0.5, aExtra.y);

      vec3 neighborRgb = mix(aColor.rgb, aBase, uSourceColor);
      float bgAlpha = mix(0.52, uDimAlpha, uDim);

      vec3 rgb = mix(aBase, neighborRgb, neighbor * uHi);
      float alpha = mix(bgAlpha, aColor.a, neighbor * uHi);
      float sizePt = mix(uBaseSize, mix(uBaseSize * 0.9, aExtra.x, uHi), neighbor);
      gl_PointSize = clamp(sizePt * uSizeScale * (15.0 / w), 1.5, 256.0);
      vSize = gl_PointSize;
      float d = distance(aPos, uEye);
      float nearB = max(uDist - uRadius, 0.0);
      float tRel = clamp((d - nearB) / max(2.0 * uRadius, 0.001), 0.0, 1.0);
      float tAbs = clamp((d - nearB) / max(uRadius * 0.8, 0.001), 0.0, 1.0);
      float fogT = uFade * (0.7 * smoothstep(0.05, 0.95, tRel) + 0.3 * smoothstep(0.0, 0.7, tAbs));
      fogT *= mix(1.0, 0.3, step(0.99, alpha));
      vFog = fogT * 0.85;
      vColor = vec4(rgb, alpha * (1.0 - fogT * fogT));
      vRing = step(1.5, aExtra.y) * uHi; // flag 2 = shared across slots
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
      float aa = clamp(2.5 / vSize, 0.02, 0.28);
      float edge = 1.0 - smoothstep(1.0 - aa, 1.0, r);
      vec3 col = vColor.rgb;
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
    attribute vec3 aPos; attribute vec4 aColor;
    uniform mat4 uViewProj; varying vec4 vColor;
    void main() { gl_Position = uViewProj * vec4(aPos, 1.0); vColor = aColor; }
  `;
  const LINE_FS = `
    precision mediump float; varying vec4 vColor; uniform float uAlpha;
    void main() { float a = vColor.a * uAlpha; gl_FragColor = vec4(vColor.rgb * a, a); }
  `;

  const renderer = {
    gl: null, pointProgram: null, lineProgram: null,
    posBuffer: null, baseColorBuffer: null, styleBuffer: null,
    gridBuffer: null, gridVertexCount: 0, dynBuffer: null, dynVertexCount: 0,
    loc: {},

    init() {
      const gl = dom.glCanvas.getContext('webgl', { antialias: true, alpha: true, premultipliedAlpha: true });
      if (!gl) { setStatus('WebGL not available.'); return; }
      this.gl = gl;
      this.pointProgram = createProgram(gl, POINT_VS, POINT_FS);
      this.lineProgram = createProgram(gl, LINE_VS, LINE_FS);
      this.loc = {
        pPos: gl.getAttribLocation(this.pointProgram, 'aPos'),
        pColor: gl.getAttribLocation(this.pointProgram, 'aColor'),
        pExtra: gl.getAttribLocation(this.pointProgram, 'aExtra'),
        pBase: gl.getAttribLocation(this.pointProgram, 'aBase'),
        pViewProj: gl.getUniformLocation(this.pointProgram, 'uViewProj'),
        pEye: gl.getUniformLocation(this.pointProgram, 'uEye'),
        pDist: gl.getUniformLocation(this.pointProgram, 'uDist'),
        pSizeScale: gl.getUniformLocation(this.pointProgram, 'uSizeScale'),
        pDim: gl.getUniformLocation(this.pointProgram, 'uDim'),
        pHi: gl.getUniformLocation(this.pointProgram, 'uHi'),
        pBaseSize: gl.getUniformLocation(this.pointProgram, 'uBaseSize'),
        pFade: gl.getUniformLocation(this.pointProgram, 'uFade'),
        pRadius: gl.getUniformLocation(this.pointProgram, 'uRadius'),
        pSourceColor: gl.getUniformLocation(this.pointProgram, 'uSourceColor'),
        pDimAlpha: gl.getUniformLocation(this.pointProgram, 'uDimAlpha'),
        lPos: gl.getAttribLocation(this.lineProgram, 'aPos'),
        lColor: gl.getAttribLocation(this.lineProgram, 'aColor'),
        lViewProj: gl.getUniformLocation(this.lineProgram, 'uViewProj'),
        lAlpha: gl.getUniformLocation(this.lineProgram, 'uAlpha'),
      };
      this.posBuffer = gl.createBuffer();
      this.baseColorBuffer = gl.createBuffer();
      this.styleBuffer = gl.createBuffer();
      this.gridBuffer = gl.createBuffer();
      this.dynBuffer = gl.createBuffer();
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.clearColor(0, 0, 0, 0);
      this.buildGrid();
    },

    buildGrid() {
      const gl = this.gl;
      const v = [];
      const grid = hexToRgb('#243549');
      const push = (x1, y1, z1, x2, y2, z2, c, a) => {
        v.push(x1, y1, z1, c[0], c[1], c[2], a, x2, y2, z2, c[0], c[1], c[2], a);
      };
      for (let i = -GRID_EXTENT; i <= GRID_EXTENT; i++) {
        if (i === 0) continue;
        push(-GRID_EXTENT, 0, i, GRID_EXTENT, 0, i, grid, 0.5);
        push(i, 0, -GRID_EXTENT, i, 0, GRID_EXTENT, grid, 0.5);
      }
      push(-GRID_EXTENT, 0, 0, GRID_EXTENT, 0, 0, hexToRgb('#ff8db3'), 0.85);
      push(0, -GRID_EXTENT, 0, 0, GRID_EXTENT, 0, hexToRgb('#8de28f'), 0.85);
      push(0, 0, -GRID_EXTENT, 0, 0, GRID_EXTENT, hexToRgb('#78e0ff'), 0.85);
      const data = new Float32Array(v);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      this.gridVertexCount = data.length / 7;
    },

    uploadCloud() {
      const gl = this.gl;
      if (!gl) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, state.positions, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.baseColorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, state.baseColors, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.styleBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, state.count * 6 * 4, gl.DYNAMIC_DRAW);
    },
    uploadStyles(arr) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.styleBuffer);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, arr);
    },
    uploadDynamicLines(data) {
      const gl = this.gl;
      this.dynVertexCount = data.length / 7;
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
      this.drawLines(this.dynBuffer, this.dynVertexCount);
      if (!state.count) return;

      gl.useProgram(this.pointProgram);
      gl.uniformMatrix4fv(this.loc.pViewProj, false, viewProj);
      gl.uniform3f(this.loc.pEye, eye[0], eye[1], eye[2]);
      gl.uniform1f(this.loc.pDist, camera.distance);
      gl.uniform1f(this.loc.pSizeScale, state.dpr * state.pointScale);
      // two blends: uDim (background) and uHi (neighbors). Both 0 = the resting
      // cloud is untouched — no dimming, no recolor, period.
      gl.uniform1f(this.loc.pDim, ease(dimA.value));
      gl.uniform1f(this.loc.pHi, ease(hi.value));
      gl.uniform1f(this.loc.pBaseSize, BASE_POINT_SIZE);
      gl.uniform1f(this.loc.pFade, state.depthFade);
      gl.uniform1f(this.loc.pRadius, state.cloudRadius);
      gl.uniform1f(this.loc.pSourceColor, state.sourceColorNodes ? 1 : 0);
      gl.uniform1f(this.loc.pDimAlpha, 0.52 * Math.pow(1 - state.dimming, 1.5));

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
      gl.drawArrays(gl.POINTS, 0, state.count);
    },
  };

  function createProgram(gl, vs, fs) {
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  // ---------------------------------------------------------------- selection animation

  // ---- selection animation, single source of truth --------------------------
  //
  // Two independent eased blends drive the two shader uniforms:
  //   dim : background dimming        0 = flat resting, 1 = dimmed
  //   hi  : neighbor highlight-in     0 = hidden,       1 = fully shown
  //
  // Splitting them is what lets a *swap* (re-query / morph scrub while already
  // selected) re-animate the neighbors (hi: 1->0->1) without the background
  // flashing back up (dim stays pinned at 1). applyStyles bakes only per-point
  // role; these blends do all the animating, so we never re-bake to animate.
  //
  const SEL_DUR = 0.55;
  const dimA = { value: 0, target: 0 };
  const hi = { value: 0, target: 0 };
  const ease = (v) => { const x = clamp(v, 0, 1); return x * x * (3 - 2 * x); };
  const hasSelection = () => state.display.some((g) => g.neighbors && g.neighbors.length);
  const animating = () => dimA.value !== dimA.target || hi.value !== hi.target;

  function stepBlend(b, dt) {
    const s = dt / SEL_DUR;
    b.value = b.target > b.value ? Math.min(b.target, b.value + s) : Math.max(b.target, b.value - s);
  }

  const styleArr = () => state._styleArr || (state._styleArr = new Float32Array(state.count * 6));

  // Bake per-point role into the style buffer. aExtra.y: 0 background, 1
  // neighbor, 2 shared across slots. aColor: slot color + target alpha/size.
  function applyStyles() {
    if (!state.count) return;
    const arr = styleArr();
    const base = state.baseColors;

    const owner = new Map(); // point index -> {rgb, count, rank, total}
    state.display.forEach((slot) => {
      if (!slot.neighbors) return;
      const rgb = hexToRgb(slot.color);
      slot.neighbors.forEach((n, rank) => {
        const prev = owner.get(n.index);
        if (prev) prev.count++;
        else owner.set(n.index, { rgb, count: 1, rank, total: slot.neighbors.length });
      });
    });

    for (let i = 0; i < state.count; i++) {
      const o = i * 6;
      const hit = owner.get(i);
      if (hit) {
        arr[o] = hit.rgb[0]; arr[o + 1] = hit.rgb[1]; arr[o + 2] = hit.rgb[2];
        arr[o + 3] = 1;                                   // neighbor target alpha
        arr[o + 4] = 4.4 - 1.3 * (hit.rank / Math.max(1, hit.total));
        arr[o + 5] = hit.count > 1 ? 2 : 1;               // role: neighbor / shared
      } else {
        arr[o] = base[i * 3]; arr[o + 1] = base[i * 3 + 1]; arr[o + 2] = base[i * 3 + 2];
        arr[o + 3] = 1; arr[o + 4] = BASE_POINT_SIZE * 0.9; arr[o + 5] = 0; // background
      }
    }
    renderer.uploadStyles(arr);
  }

  // Fresh selection from a resting cloud: dim the background AND fade neighbors in.
  function showSelection() {
    applyStyles();
    updateResults();
    dimA.target = 1;
    hi.value = 0; hi.target = 1; // always replay the highlight-in
    state.needsRender = true;
  }

  // Swap while already selected (re-query, morph scrub): keep the background
  // dimmed (dimA pinned), re-animate the neighbors in (hi 0 -> 1).
  function swapSelection() {
    applyStyles();
    updateResults();
    dimA.value = 1; dimA.target = 1;
    hi.value = 0; hi.target = 1;
    state.needsRender = true;
  }

  // Animate the whole selection out; frame() drops the slots when both hit 0.
  function dismissSelection() {
    if (!hasSelection()) return;
    if (state.playing) stopAutoplay();
    dimA.target = 0; hi.target = 0;
    state.needsRender = true;
  }

  function resetSelectionNow() {
    state.slots.forEach((s) => { s.ghost = null; s.neighbors = null; s.vec = null; });
    state.display = [];
    dom.morph.hidden = true; state.morph = null;
    dimA.value = dimA.target = 0; hi.value = hi.target = 0;
    applyStyles(); rebuildLines(); updateResults();
    state.needsRender = true;
  }

  function rebuildLines(growth = 1) {
    const v = [];
    const push = (a, b, rgb, alpha, g = growth) => {
      if (g <= 0) return;
      const ex = a[0] + (b[0] - a[0]) * g, ey = a[1] + (b[1] - a[1]) * g, ez = a[2] + (b[2] - a[2]) * g;
      v.push(a[0], a[1], a[2], rgb[0], rgb[1], rgb[2], alpha, ex, ey, ez, rgb[0], rgb[1], rgb[2], alpha);
    };
    const at = (i) => [state.positions[i * 3], state.positions[i * 3 + 1], state.positions[i * 3 + 2]];
    for (const slot of state.display) {
      if (!slot.neighbors || !slot.ghost) continue;
      const rgb = hexToRgb(slot.color);
      for (const n of slot.neighbors) push(slot.ghost, at(n.index), rgb, 0.22);
    }
    renderer.uploadDynamicLines(new Float32Array(v));
  }

  // ---------------------------------------------------------------- camera (from word app)

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
      for (let i = 0; i < 3; i++) camera.target[i] = lerp(tw.fromTarget[i], tw.toTarget[i], e);
      if (tw.t >= 1) camera.tween = null;
      active = true;
    }
    if (!pointerState.dragging && !camera.tween) {
      if (Math.abs(camera.velYaw) > 0.0004 || Math.abs(camera.velPitch) > 0.0004) {
        camera.yaw += camera.velYaw * dt;
        camera.pitch = clamp(camera.pitch + camera.velPitch * dt, -1.45, 1.45);
        const decay = Math.exp(-dt * 4.6);
        camera.velYaw *= decay; camera.velPitch *= decay;
        active = true;
      } else { camera.velYaw = 0; camera.velPitch = 0; }
    }
    const dd = camera.distanceTarget - camera.distance;
    if (Math.abs(dd) > 0.001) { camera.distance += dd * (1 - Math.exp(-dt * 9)); active = true; }

    // follow-during-scrub: spring the target toward the live follow point.
    // No tween restart per tick, so it never lags behind — and the spring
    // constant scales with how far behind it is, so fast scrubbing catches up
    // fast while slow scrubbing stays smooth.
    if (camera.follow && !camera.tween && !pointerState.dragging) {
      const f = camera.follow, t = camera.target;
      const gap = Math.hypot(f[0] - t[0], f[1] - t[1], f[2] - t[2]);
      if (gap > 0.002) {
        // base rate ~10/s, boosted up to ~28/s when far behind (gap ~ cloud size)
        const rate = 10 + 18 * clamp(gap / (state.cloudRadius || 4.4), 0, 1);
        const k = 1 - Math.exp(-dt * rate);
        t[0] += (f[0] - t[0]) * k; t[1] += (f[1] - t[1]) * k; t[2] += (f[2] - t[2]) * k;
        active = true;
      }
    }
    return active;
  }
  function startCameraTween(toYaw, toPitch, toDist, toTarget, duration = 0.6) {
    camera.velYaw = 0; camera.velPitch = 0;
    camera.tween = {
      t: 0, duration, fromYaw: camera.yaw, toYaw, fromPitch: camera.pitch, toPitch,
      fromDist: camera.distance, toDist, fromTarget: camera.target.slice(), toTarget: toTarget.slice(),
    };
  }
  function resetCamera() {
    camera.follow = null;
    startCameraTween(CAMERA_DEFAULTS.yaw, CAMERA_DEFAULTS.pitch, CAMERA_DEFAULTS.distance, CAMERA_DEFAULTS.target);
  }

  function computeMatrices() {
    cameraEye();
    const aspect = Math.max(0.1, state.cssWidth / Math.max(1, state.cssHeight));
    mat4Perspective(projMatrix, Math.PI / 4, aspect, 0.1, 300);
    mat4LookAt(viewMatrix, eye, camera.target, [0, 1, 0]);
    mat4Multiply(viewProj, projMatrix, viewMatrix);
  }
  function computeScreenPositions() {
    if (!screenXY) return;
    const m = viewProj, pos = state.positions, w = state.cssWidth, h = state.cssHeight;
    for (let i = 0; i < state.count; i++) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (cw < 0.02) { screenXY[i * 3 + 2] = -1; continue; }
      screenXY[i * 3] = ((m[0] * x + m[4] * y + m[8] * z + m[12]) / cw * 0.5 + 0.5) * w;
      screenXY[i * 3 + 1] = (0.5 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw * 0.5) * h;
      screenXY[i * 3 + 2] = cw;
    }
  }

  let lastFrameTime = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - lastFrameTime) / 1000);
    lastFrameTime = now;
    if (animating()) {
      stepBlend(dimA, dt);
      stepBlend(hi, dt);
      rebuildLines(ease(hi.value)); // lines grow with the neighbor highlight
      if (dimA.value === 0 && dimA.target === 0 && hi.value === 0 && hi.target === 0) {
        // fully dismissed: drop the selection, settle at rest
        state.slots.forEach((s) => { s.ghost = null; s.neighbors = null; s.vec = null; });
        state.display = [];
        dom.morph.hidden = true; state.morph = null;
        applyStyles(); updateResults();
      }
      state.needsRender = true;
    }
    const camActive = updateCamera(dt);
    if (camActive || state.needsRender) {
      computeMatrices(); renderer.draw(); computeScreenPositions();
      state.needsRender = false;
    }
    drawOverlay();
    requestAnimationFrame(frame);
  }

  // ---------------------------------------------------------------- overlay

  function pointScreenRadius(index) {
    const w = screenXY[index * 3 + 2];
    if (w <= 0) return 0;
    const arr = styleArr();
    const size = arr[index * 6 + 4] || BASE_POINT_SIZE;
    return clamp(size * state.pointScale * (15 / w) * 0.5, 1, 128);
  }

  function drawOverlay() {
    octx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    octx.clearRect(0, 0, state.cssWidth, state.cssHeight);
    if (!state.count || !screenXY) return;

    octx.globalAlpha = ease(hi.value); // ghosts fade with the neighbor highlight
    for (const slot of state.display) {
      if (!slot.ghost || !slot.neighbors) continue;
      if (!projectToScreen(slot.ghost[0], slot.ghost[1], slot.ghost[2], scratchPoint)) continue;
      drawGhost(scratchPoint.x, scratchPoint.y, slot.color);
    }
    octx.globalAlpha = 1;

    if (state.hoverIndex >= 0) drawHover(state.hoverIndex);
  }

  function drawGhost(x, y, color) {
    octx.save();
    octx.translate(x, y); octx.rotate(Math.PI / 4);
    const s = 6;
    octx.fillStyle = hexToCss(color, 0.98);
    octx.strokeStyle = 'rgba(6,12,21,0.8)'; octx.lineWidth = 1.6;
    octx.fillRect(-s, -s, s * 2, s * 2); octx.strokeRect(-s, -s, s * 2, s * 2);
    octx.restore();
    octx.beginPath(); octx.arc(x, y, 12, 0, Math.PI * 2);
    octx.strokeStyle = hexToCss(color, 0.4); octx.lineWidth = 1.5; octx.stroke();
  }

  function drawHover(index) {
    if (screenXY[index * 3 + 2] <= 0) return;
    const x = screenXY[index * 3], y = screenXY[index * 3 + 1];
    const r = pointScreenRadius(index) + 4;
    octx.beginPath(); octx.arc(x, y, r, 0, Math.PI * 2);
    octx.strokeStyle = 'rgba(255,255,255,0.85)'; octx.lineWidth = 1.6; octx.stroke();

    const src = state.sources[state.sourceIdx[index]];
    const text = passageText(index);
    drawTooltip(x + r + 8, y, text, src);
  }

  function drawTooltip(x, y, text, src) {
    octx.font = '12px "IBM Plex Mono", monospace';
    const maxW = 320;
    const words = text.split(' ');
    const lines = [];
    let line = '';
    for (const word of words) {
      const test = line ? line + ' ' + word : word;
      if (octx.measureText(test).width > maxW && line) { lines.push(line); line = word; }
      else line = test;
      if (lines.length >= 4) break;
    }
    if (line && lines.length < 4) lines.push(line);
    const badge = src ? src.name : '';
    const lh = 17, pad = 9;
    const boxW = maxW + pad * 2;
    const boxH = lines.length * lh + 20 + pad * 2;
    let bx = x, by = y - boxH / 2;
    if (bx + boxW > state.cssWidth - 6) bx = x - boxW - (pointScreenRadius(state.hoverIndex) + 16);
    bx = Math.max(6, bx);
    by = clamp(by, 6, state.cssHeight - boxH - 6);

    octx.fillStyle = 'rgba(6,12,21,0.94)';
    octx.strokeStyle = 'rgba(255,255,255,0.14)'; octx.lineWidth = 1;
    roundRect(bx, by, boxW, boxH, 9); octx.fill(); octx.stroke();

    if (src) {
      octx.fillStyle = hexToCss(src.color, 1);
      octx.beginPath(); octx.arc(bx + pad + 4, by + pad + 6, 4, 0, Math.PI * 2); octx.fill();
      octx.fillStyle = hexToCss(src.color, 0.95);
      octx.font = '11px "IBM Plex Mono", monospace';
      octx.textBaseline = 'middle';
      octx.fillText(badge, bx + pad + 14, by + pad + 6);
    }
    octx.fillStyle = '#ecf4ff';
    octx.font = '12px "IBM Plex Mono", monospace';
    octx.textBaseline = 'top';
    lines.forEach((ln, i) => octx.fillText(ln, bx + pad, by + pad + 18 + i * lh));
  }

  function roundRect(x, y, w, h, r) {
    octx.beginPath();
    octx.moveTo(x + r, y);
    octx.arcTo(x + w, y, x + w, y + h, r);
    octx.arcTo(x + w, y + h, x, y + h, r);
    octx.arcTo(x, y + h, x, y, r);
    octx.arcTo(x, y, x + w, y, r);
    octx.closePath();
  }

  function pickPointAt(x, y) {
    if (!screenXY) return -1;
    let best = -1, bestW = Infinity;
    const arr = styleArr();
    for (let i = 0; i < state.count; i++) {
      const w = screenXY[i * 3 + 2];
      if (w <= 0 || arr[i * 6 + 3] < 0.03) continue;
      const dx = screenXY[i * 3] - x, dy = screenXY[i * 3 + 1] - y;
      if (Math.hypot(dx, dy) < Math.max(7, pointScreenRadius(i) + 3) && w < bestW) { bestW = w; best = i; }
    }
    return best;
  }

  // ghost markers (query/morph result) are bigger click targets than points
  function pickGhostAt(x, y) {
    for (const slot of state.display) {
      if (!slot.ghost || !slot.neighbors) continue;
      if (!projectToScreen(slot.ghost[0], slot.ghost[1], slot.ghost[2], scratchPoint)) continue;
      if (Math.hypot(scratchPoint.x - x, scratchPoint.y - y) < 16) return slot.ghost.slice();
    }
    return null;
  }

  function centerCameraOn(worldPos) {
    camera.follow = null; // a deliberate click overrides scrub-follow
    startCameraTween(camera.yaw, camera.pitch, Math.min(camera.distance, 7), worldPos, 0.5);
  }

  // ---------------------------------------------------------------- embedding (transformers.js)

  const extractors = new Map();      // browserId -> extractor
  const extractorLoading = new Map(); // browserId -> promise

  async function ensureExtractor(browserId = state.browserId) {
    if (extractors.has(browserId)) return extractors.get(browserId);
    if (extractorLoading.has(browserId)) return extractorLoading.get(browserId);
    const item = addLoadingItem('embedding model', 0);
    const promise = pipeline('feature-extraction', browserId, {
      progress_callback: (p) => {
        if (p.status === 'progress' && p.total) item.update(p.loaded / p.total, p.loaded / 1e6, p.total / 1e6);
      },
    }).then((ex) => { extractors.set(browserId, ex); item.done(); return ex; });
    extractorLoading.set(browserId, promise);
    return promise;
  }

  async function embedText(text) {
    const ex = await ensureExtractor();
    const out = await ex(text, { pooling: 'mean', normalize: true });
    return new Float32Array(out.data);
  }

  // ---------------------------------------------------------------- search worker

  const search = { worker: null, ready: false, nextId: 1, pending: new Map() };

  function initWorker(precision, codesBuffer, scalesBuffer) {
    if (search.worker) search.worker.terminate();
    search.ready = false; search.pending.clear();
    search.worker = new Worker('search-worker.js');
    search.worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'ready') { search.ready = true; return; }
      if (m.type === 'result') {
        const r = search.pending.get(m.id);
        if (r) { search.pending.delete(m.id); r(m.neighbors); }
      }
    };
    const transfer = [codesBuffer];
    if (scalesBuffer) transfer.push(scalesBuffer);
    search.worker.postMessage({ type: 'init', precision, dim: state.dim, count: state.count, codes: codesBuffer, scales: scalesBuffer }, transfer);
  }

  function searchNeighbors(vector, topN, excludeSet) {
    return new Promise((resolve) => {
      if (!search.worker || !search.ready) { resolve([]); return; }
      const id = search.nextId++;
      search.pending.set(id, resolve);
      const copy = new Float32Array(vector);
      search.worker.postMessage({ type: 'query', id, vector: copy.buffer, topN, exclude: [...excludeSet] }, [copy.buffer]);
    });
  }

  // ---------------------------------------------------------------- data loading

  function setStatus(msg) { dom.statusBar.textContent = msg; }

  function addLoadingItem(label, totalMb) {
    dom.loadingPanel.classList.add('active');
    const li = document.createElement('li');
    li.innerHTML = `<span class="li-label">${label}</span><span class="li-bar"><i></i></span><span class="li-num"></span>`;
    dom.loadingItems.appendChild(li);
    const bar = li.querySelector('i'), num = li.querySelector('.li-num');
    return {
      update(frac, mb, total) {
        bar.style.width = `${Math.round(frac * 100)}%`;
        num.textContent = `${mb.toFixed(1)} / ${(total || totalMb).toFixed(0)} MB`;
      },
      done() { bar.style.width = '100%'; num.textContent = 'done'; li.classList.add('li-done'); maybeHidePanel(); },
    };
  }
  function maybeHidePanel() {
    if ([...dom.loadingItems.children].every((li) => li.classList.contains('li-done'))) {
      setTimeout(() => dom.loadingPanel.classList.remove('active'), 500);
    }
  }

  async function fetchBinary(url, onProgress) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`${url}: ${resp.status}`);
    const total = Number(resp.headers.get('Content-Length')) || 0;
    if (!resp.body || !total || !onProgress) return resp.arrayBuffer();
    const reader = resp.body.getReader();
    const chunks = []; let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); received += value.length;
      onProgress(received, total);
    }
    const buf = new Uint8Array(received); let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    return buf.buffer;
  }

  function normalizeProjection() {
    const pos = state.positions, count = state.count;
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < count; i++) { cx += pos[i * 3]; cy += pos[i * 3 + 1]; cz += pos[i * 3 + 2]; }
    cx /= count; cy /= count; cz /= count;
    const radii = new Float32Array(count);
    for (let i = 0; i < count; i++) radii[i] = Math.hypot(pos[i * 3] - cx, pos[i * 3 + 1] - cy, pos[i * 3 + 2] - cz);
    const p95 = radii.slice().sort()[Math.floor(count * 0.95)] || 1;
    const scale = 4.4 / p95;
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (pos[i * 3] - cx) * scale;
      pos[i * 3 + 1] = (pos[i * 3 + 1] - cy) * scale;
      pos[i * 3 + 2] = (pos[i * 3 + 2] - cz) * scale;
    }
    state.cloudRadius = 4.4;
  }

  function buildBaseColors() {
    state.baseColors = new Float32Array(state.count * 3);
    const rgbs = state.sources.map((s) => hexToRgb(s.color));
    for (let i = 0; i < state.count; i++) {
      const c = rgbs[state.sourceIdx[i]] || [0.5, 0.55, 0.65];
      state.baseColors[i * 3] = c[0]; state.baseColors[i * 3 + 1] = c[1]; state.baseColors[i * 3 + 2] = c[2];
    }
  }

  let index = null;
  let loadingData = false;

  async function loadTier(size, model, precision) {
    if (loadingData) return;
    loadingData = true;
    dom.modelSelect.disabled = dom.datasetSelect.disabled = dom.precisionSelect.disabled = true;
    try {
      const tierDir = `data/${size}`;
      const modelDir = `${tierDir}/${model}`;
      const manifest = await (await fetch(`${modelDir}/manifest.json`)).json();
      state.count = manifest.count;
      state.dim = manifest.dim;
      state.sources = manifest.sources;
      state.model = model;
      state.precision = precision;
      state.tierSize = size;
      const modelMeta = index.models.find((m) => m.key === model);
      state.browserId = modelMeta ? modelMeta.browserId : state.browserId;

      const embItem = addLoadingItem(`corpus (${size.toLocaleString()}, ${model}/${precision})`, 0);
      const [projBuf, srcBuf, textBuf, offBuf, embBuf, scaleBuf] = await Promise.all([
        fetchBinary(`${modelDir}/projected.bin`),
        fetchBinary(`${tierDir}/sources.bin`),
        fetchBinary(`${tierDir}/texts.bin`),
        fetchBinary(`${tierDir}/text_offsets.bin`),
        fetchBinary(`${modelDir}/${precision}/embeddings.bin`, (l, t) => embItem.update(l / t, l / 1e6, t / 1e6)),
        precision === 'f32' ? Promise.resolve(null) : fetchBinary(`${modelDir}/${precision}/scales.bin`),
      ]);
      embItem.done();

      state.positions = new Float32Array(projBuf);
      state.sourceIdx = new Uint8Array(srcBuf);
      state.textBlob = new Uint8Array(textBuf);
      state.textOffsets = new Uint32Array(offBuf);
      state._styleArr = null;
      screenXY = new Float32Array(state.count * 3);

      normalizeProjection();
      buildBaseColors();
      renderer.uploadCloud();
      initWorker(precision, embBuf, scaleBuf);
      buildLegend();
      resetSelectionNow();
      state.needsRender = true;

      // prefetch the model's ONNX weights so the first query is warm
      ensureExtractor(state.browserId);

      syncSelectors();
      setStatus(`${state.count.toLocaleString()} passages · ${model}. Type two phrasings and hit Run.`);
    } catch (err) {
      setStatus('Failed to load corpus.');
      console.error(err);
    } finally {
      loadingData = false;
      dom.modelSelect.disabled = dom.datasetSelect.disabled = dom.precisionSelect.disabled = false;
    }
  }

  function buildLegend() {
    dom.legend.innerHTML = '';
    for (const s of state.sources) {
      const chip = document.createElement('span');
      chip.className = 'legend-chip';
      chip.innerHTML = `<i style="background:${s.color}"></i>${s.name}`;
      dom.legend.appendChild(chip);
    }
  }

  const option = (value, label) => {
    const o = document.createElement('option');
    o.value = value; o.textContent = label;
    return o;
  };
  const tierEntry = (size) => index.tiers.find((t) => t.size === size);
  const precisionsFor = (size, model) => (tierEntry(size).models[model] || []);

  function populateSelectors() {
    dom.modelSelect.innerHTML = '';
    for (const m of index.models) dom.modelSelect.appendChild(option(m.key, m.label));
    dom.datasetSelect.innerHTML = '';
    for (const tier of index.tiers) {
      dom.datasetSelect.appendChild(option(String(tier.size), `${tier.size.toLocaleString()} passages`));
    }
    dom.modelSelect.hidden = dom.datasetSelect.hidden = dom.precisionSelect.hidden = false;
  }

  // keep the three selectors mutually consistent with what actually exists
  function syncSelectors() {
    dom.modelSelect.value = state.model;
    dom.datasetSelect.value = String(state.tierSize);
    const order = ['int8', 'int16', 'f32'];
    const avail = precisionsFor(state.tierSize, state.model);
    dom.precisionSelect.innerHTML = '';
    for (const p of order) if (avail.includes(p)) dom.precisionSelect.appendChild(option(p, p));
    dom.precisionSelect.value = avail.includes(state.precision) ? state.precision : avail[0];
    state.precision = dom.precisionSelect.value;
  }

  function pickPrecision(size, model) {
    const avail = precisionsFor(size, model);
    return avail.includes(state.precision) ? state.precision : (avail.includes('int8') ? 'int8' : avail[0]);
  }

  // ---------------------------------------------------------------- slots / query

  function addSlot(text = '') {
    if (state.slots.length >= SLOT_COLORS.length) return;
    const color = SLOT_COLORS[state.slots.length];
    const slot = { text, color, ghost: null, neighbors: null };
    state.slots.push(slot);
    renderSlots();
  }

  function renderSlots() {
    dom.slots.innerHTML = '';
    state.slots.forEach((slot, i) => {
      const row = document.createElement('div');
      row.className = 'slot-row';
      row.innerHTML = `
        <input type="color" value="${slot.color}" aria-label="Slot color">
        <input type="text" class="slot-input" placeholder="phrasing ${i + 1}" value="${slot.text.replace(/"/g, '&quot;')}">
        ${state.slots.length > 1 ? '<button type="button" class="slot-del" aria-label="Remove">×</button>' : ''}
      `;
      const [colorInput, textInput] = row.querySelectorAll('input');
      colorInput.addEventListener('input', () => {
        slot.color = colorInput.value;
        // reflect the recolor in the live display (unless a morph group is up)
        const d = state.display.find((g) => g.text === slot.text);
        if (d) d.color = slot.color;
        applyStyles(); rebuildLines(); updateResults(); state.needsRender = true;
      });
      textInput.addEventListener('input', () => { slot.text = textInput.value; });
      textInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') runAll(); });
      const del = row.querySelector('.slot-del');
      if (del) del.addEventListener('click', () => { state.slots.splice(i, 1); renderSlots(); runAll(); });
      dom.slots.appendChild(row);
    });
  }

  async function runAll() {
    const active = state.slots.filter((s) => s.text.trim());
    if (!active.length) { setStatus('Type at least one phrasing.'); return; }
    if (!search.ready) { setStatus('Corpus still loading…'); return; }
    if (state.playing) stopAutoplay();

    // already dimmed on screen? swap (keep bg dimmed, re-animate neighbors);
    // else play the full dim-in. Keys off the actual blend, so it survives
    // slots being reset by example chips / morph.
    const wasShowing = dimA.value > 0.5 && dimA.target === 1;
    dom.runButton.disabled = true;
    setStatus('Embedding in your browser…');
    try {
      for (const slot of state.slots) {
        if (!slot.text.trim()) { slot.ghost = null; slot.neighbors = null; slot.vec = null; continue; }
        const vec = await embedText(slot.text.trim());
        slot.vec = vec;
        slot.neighbors = await searchNeighbors(vec, state.topN, new Set());
        slot.ghost = ghostPosition(slot.neighbors);
      }
      // display mirrors the active query slots (morph will later swap in its
      // own single group without touching state.slots)
      state.display = state.slots
        .filter((s) => s.neighbors && s.neighbors.length)
        .map((s) => ({ color: s.color, neighbors: s.neighbors, ghost: s.ghost, text: s.text }));
      if (wasShowing) swapSelection();
      else showSelection();
      setupMorph(); // always rebuild the morph for the new pair (fresh slider + cache)
      setStatus(`Compared ${active.length} phrasing${active.length > 1 ? 's' : ''}.`);
    } catch (err) {
      setStatus('Embedding failed — see console.');
      console.error(err);
    } finally {
      dom.runButton.disabled = false;
    }
  }

  function ghostPosition(neighbors) {
    const pos = state.positions;
    const k = Math.min(6, neighbors.length);
    if (!k) return null;
    let wx = 0, wy = 0, wz = 0, ws = 0;
    for (let i = 0; i < k; i++) {
      const n = neighbors[i], w = Math.pow(Math.max(n.score, 0) + 0.01, 6);
      wx += pos[n.index * 3] * w; wy += pos[n.index * 3 + 1] * w; wz += pos[n.index * 3 + 2] * w; ws += w;
    }
    return ws ? [wx / ws, wy / ws, wz / ws] : null;
  }

  // Morph: interpolate between two phrasing vectors and watch the neighborhood
  // travel. The whole point of the app made kinetic — the throwaround slider.
  const MORPH_STEPS = 100; // slider is 0..100
  const morphDebounce = { timer: 0 };

  function setupMorph() {
    if (state.playing) stopAutoplay();       // don't let a new pair replay the old sweep
    play.t = 0; play.lastStep = -1;
    morphDebounce.busy = false; morphDebounce.pending = undefined;
    const active = state.slots.filter((s) => s.vec && s.neighbors);
    if (active.length !== 2) { dom.morph.hidden = true; state.morph = null; return; }
    // new token space + null cache invalidates any in-flight precompute/search
    state.morph = { a: active[0], b: active[1], token: 0, cache: null };
    dom.morph.hidden = false;
    dom.morphSlider.value = '0';
    updateMorphLabel(0);
    refreshMorphControls();
    if (state.precise) precomputeMorph();
  }

  // play button + animate-steps toggle only make sense once precise is cached
  function refreshMorphControls() {
    const ready = state.precise && state.morph && state.morph.cache;
    dom.playButton.hidden = !ready;
  }

  function updateMorphLabel(t) {
    const m = state.morph;
    if (!m) return;
    const pct = Math.round(t * 100);
    const mode = state.precise ? (m.cache ? '· precise' : '· precomputing…') : '';
    dom.morphLabel.innerHTML =
      `<span style="color:${m.a.color}">A</span> <span class="morph-pct">${100 - pct}% · ${pct}%</span> <span style="color:${m.b.color}">B</span> <span class="morph-mode">${mode}</span>`;
  }

  function morphVector(t) {
    const m = state.morph;
    const dim = m.a.vec.length;
    const v = new Float32Array(dim);
    let norm = 0;
    for (let i = 0; i < dim; i++) { v[i] = lerp(m.a.vec[i], m.b.vec[i], t); norm += v[i] * v[i]; }
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < dim; i++) v[i] /= norm;
    return v;
  }

  // Paint one morph step. Morph is a continuous scrub, not a discrete select,
  // so it never plays the highlight-in animation — the point of the slider is
  // that the neighborhood slides smoothly with your thumb.
  function paintMorphStep(t, neighbors) {
    const m = state.morph;
    const morphColor = mixHex(m.a.color, m.b.color, t);
    const ghost = ghostPosition(neighbors);
    // render into DISPLAY only — never touch state.slots (the user's phrasings)
    state.display = [{ color: morphColor, neighbors, ghost, text: `morph ${Math.round(t * 100)}%` }];
    applyStyles();
    updateResults();
    if (state.followResult && ghost) {
      camera.follow = ghost.slice();
      camera.tween = null;
    }
    dimA.value = dimA.target = 1;
    hi.value = hi.target = 1; // pinned: no per-step fade
    rebuildLines(1);
    state.needsRender = true;
  }

  // Live mode: THROTTLE (not debounce) — fire at most every MORPH_THROTTLE ms
  // *while* dragging, so a slow-but-continuous drag still updates regularly
  // instead of freezing until you pause.
  const MORPH_THROTTLE = 200;
  function morphLive(t) {
    updateMorphLabel(t);
    morphDebounce.pending = t;
    if (morphDebounce.busy) return;
    const elapsed = performance.now() - (morphDebounce.last || 0);
    const fire = async () => {
      morphDebounce.busy = true;
      morphDebounce.last = performance.now();
      const at = morphDebounce.pending;
      const m = state.morph;
      if (!m) { morphDebounce.busy = false; return; }
      const token = ++m.token;
      const neighbors = await searchNeighbors(morphVector(at), state.topN, new Set());
      morphDebounce.busy = false;
      if (!state.morph || m.token !== token) return;
      paintMorphStep(at, neighbors);
      // if the thumb moved on during the search, schedule the latest value
      if (morphDebounce.pending !== at) morphLive(morphDebounce.pending);
    };
    if (elapsed >= MORPH_THROTTLE) fire();
    else {
      clearTimeout(morphDebounce.timer);
      morphDebounce.timer = setTimeout(fire, MORPH_THROTTLE - elapsed);
    }
  }

  // Precise mode: all steps precomputed → instant swap on every input.
  function morphPlayback(t) {
    const m = state.morph;
    if (!m || !m.cache) { morphLive(t); return; }
    updateMorphLabel(t);
    const idx = clamp(Math.round(t * MORPH_STEPS), 0, MORPH_STEPS);
    paintMorphStep(t, m.cache[idx]);
  }

  function morphTo(t) {
    if (state.precise && state.morph && state.morph.cache) morphPlayback(t);
    else morphLive(t);
  }

  // Autoplay: walk the slider 0 -> 100% hands-free, animating each step.
  const play = { raf: 0, t: 0, lastStep: -1 };
  function toggleAutoplay() {
    if (state.playing) { stopAutoplay(); return; }
    const m = state.morph;
    if (!m) return;
    if (!state.precise || !m.cache) { setStatus('Enable "precise animation" to autoplay.'); return; }
    state.playing = true;
    dom.playButton.textContent = '❚❚ pause';
    play.t = 0; play.lastStep = -1;
    let last = performance.now();
    const tick = (now) => {
      if (!state.playing) return;
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      play.t = Math.min(1, play.t + dt / 6); // full sweep in ~6s
      dom.morphSlider.value = String(Math.round(play.t * 100));
      const step = Math.round(play.t * MORPH_STEPS);
      if (step !== play.lastStep) {
        play.lastStep = step;
        morphPlayback(play.t); // precise cache guaranteed (checked before start)
      }
      if (play.t >= 1) { stopAutoplay(); return; }
      play.raf = requestAnimationFrame(tick);
    };
    play.raf = requestAnimationFrame(tick);
  }
  function stopAutoplay() {
    state.playing = false;
    cancelAnimationFrame(play.raf);
    dom.playButton.textContent = '▶ play';
  }

  async function precomputeMorph() {
    const m = state.morph;
    if (!m) return;
    m.cache = null;
    updateMorphLabel(Number(dom.morphSlider.value) / 100);
    setStatus('Precomputing morph path…');
    const token = ++m.token;
    const cache = new Array(MORPH_STEPS + 1);
    for (let i = 0; i <= MORPH_STEPS; i++) {
      const neighbors = await searchNeighbors(morphVector(i / MORPH_STEPS), state.topN, new Set());
      if (!state.morph || m.token !== token) return; // aborted (new query / model)
      cache[i] = neighbors;
    }
    m.cache = cache;
    updateMorphLabel(Number(dom.morphSlider.value) / 100);
    refreshMorphControls();
    setStatus('Morph path ready — scrub freely, or ▶ play.');
  }

  function mixHex(h1, h2, t) {
    const a = hexToRgb(h1), b = hexToRgb(h2);
    const c = [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
    return '#' + c.map((x) => Math.round(x * 255).toString(16).padStart(2, '0')).join('');
  }

  // ---------------------------------------------------------------- results + metrics

  function cosineVecs(a, b) {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / (Math.sqrt(na) * Math.sqrt(nb) || 1);
  }

  function updateResults() {
    const active = state.display.filter((s) => s.neighbors && s.neighbors.length);
    dom.metrics.innerHTML = '';
    dom.resultList.innerHTML = '';

    if (!active.length) {
      dom.resultSummary.textContent = 'Run two phrasings to compare their neighborhoods.';
      return;
    }
    dom.resultSummary.textContent = `${active.length} phrasing${active.length > 1 ? 's' : ''} · ${state.topN} neighbors each.`;

    // rank-weighted source-mix per slot: nearer neighbors count more, so the
    // register signal (strongest in the top matches) drives the bars
    active.forEach((slot) => {
      const weight = new Array(state.sources.length).fill(0);
      let wsum = 0;
      for (const n of slot.neighbors) {
        const w = Math.pow(Math.max(n.score, 0), 2);
        weight[state.sourceIdx[n.index]] += w;
        wsum += w;
      }
      wsum = wsum || 1;
      const card = document.createElement('div');
      card.className = 'metric-card';
      card.style.borderColor = hexToCss(slot.color, 0.4);
      const bars = state.sources.map((s, si) => {
        const pct = Math.round((weight[si] / wsum) * 100);
        return `<div class="mix-row"><span>${s.name}</span><span class="mix-bar"><i style="width:${pct}%;background:${s.color}"></i></span><b>${pct}%</b></div>`;
      }).join('');
      card.innerHTML = `<div class="metric-title" style="color:${slot.color}">${escapeHtml(slot.text.slice(0, 40))}${slot.text.length > 40 ? '…' : ''}</div>${bars}`;
      dom.metrics.appendChild(card);
    });

    // pairwise similarity (via neighbor overlap — proxy for query cosine)
    if (active.length >= 2) {
      const pairWrap = document.createElement('div');
      pairWrap.className = 'metric-card pairs';
      let html = '<div class="metric-title">Neighbor overlap</div>';
      for (let i = 0; i < active.length; i++) {
        for (let j = i + 1; j < active.length; j++) {
          const a = new Set(active[i].neighbors.map((n) => n.index));
          const b = new Set(active[j].neighbors.map((n) => n.index));
          let inter = 0; for (const x of a) if (b.has(x)) inter++;
          const jac = Math.round((inter / (a.size + b.size - inter)) * 100);
          html += `<div class="mix-row"><span><i style="background:${active[i].color};display:inline-block;width:8px;height:8px;border-radius:2px"></i> ∩ <i style="background:${active[j].color};display:inline-block;width:8px;height:8px;border-radius:2px"></i></span><span class="mix-bar"><i style="width:${jac}%;background:#9db8d9"></i></span><b>${jac}%</b></div>`;
        }
      }
      pairWrap.innerHTML = html;
      dom.metrics.appendChild(pairWrap);
    }

    // neighbor list (per slot, color-tagged)
    active.forEach((slot) => {
      slot.neighbors.slice(0, 12).forEach((n) => {
        const li = document.createElement('li');
        li.className = 'result-row';
        li.style.borderLeft = `3px solid ${slot.color}`;
        const src = state.sources[state.sourceIdx[n.index]];
        li.innerHTML = `<span class="rr-src" style="color:${src.color}">${src.name}</span><span class="rr-text">${escapeHtml(passageText(n.index))}</span><span class="rr-score">${n.score.toFixed(2)}</span>`;
        li.addEventListener('mouseenter', () => { state.hoverIndex = n.index; });
        li.addEventListener('mouseleave', () => { if (state.hoverIndex === n.index) state.hoverIndex = -1; });
        li.addEventListener('click', () => {
          const p = state.positions;
          startCameraTween(camera.yaw, camera.pitch, Math.min(camera.distance, 7),
            [p[n.index * 3], p[n.index * 3 + 1], p[n.index * 3 + 2]], 0.55);
        });
        dom.resultList.appendChild(li);
      });
    });
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  // ---------------------------------------------------------------- events

  function resize() {
    const rect = dom.stage.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    if (rect.width === state.cssWidth && rect.height === state.cssHeight && dpr === state.dpr) return;
    state.cssWidth = rect.width; state.cssHeight = rect.height; state.dpr = dpr;
    const pw = Math.max(1, Math.round(rect.width * dpr)), ph = Math.max(1, Math.round(rect.height * dpr));
    dom.glCanvas.width = pw; dom.glCanvas.height = ph;
    dom.overlay.width = pw; dom.overlay.height = ph;
    state.needsRender = true;
  }

  function bindStage() {
    const stage = dom.stage;
    const overCanvas = (e) => e.target === dom.glCanvas || e.target === dom.stage || e.target === dom.overlay;
    stage.addEventListener('contextmenu', (e) => e.preventDefault());

    stage.addEventListener('pointerdown', (e) => {
      if (!overCanvas(e)) return;
      camera.follow = null; // manual control cancels result-follow
      stage.setPointerCapture(e.pointerId);
      pointerState.pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
      if (pointerState.pointers.size === 2) {
        const p = [...pointerState.pointers.values()];
        pointerState.pinchDist = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
        pointerState.mode = 'pinch'; pointerState.dragging = true; return;
      }
      camera.tween = null; camera.velYaw = camera.velPitch = 0;
      pointerState.dragging = true;
      pointerState.mode = (e.button === 2 || e.button === 1 || e.ctrlKey || e.shiftKey) ? 'pan' : 'orbit';
      pointerState.lastX = e.offsetX; pointerState.lastY = e.offsetY;
      pointerState.downX = e.offsetX; pointerState.downY = e.offsetY;
      pointerState.moved = 0; pointerState.lastMoveTime = performance.now();
      stage.classList.add('grabbing');
    });

    stage.addEventListener('pointermove', (e) => {
      const tr = pointerState.pointers.get(e.pointerId);
      if (tr) { tr.x = e.offsetX; tr.y = e.offsetY; }
      if (pointerState.mode === 'pinch' && pointerState.pointers.size === 2) {
        const p = [...pointerState.pointers.values()];
        const dist = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
        if (pointerState.pinchDist > 0 && dist > 0)
          camera.distanceTarget = clamp(camera.distanceTarget * (pointerState.pinchDist / dist), 0.2, 120);
        pointerState.pinchDist = dist; return;
      }
      if (!pointerState.dragging) {
        if (!overCanvas(e)) { state.hoverIndex = -1; stage.classList.remove('point-hover'); return; }
        const picked = pickPointAt(e.offsetX, e.offsetY);
        state.hoverIndex = picked;
        stage.classList.toggle('point-hover', picked >= 0);
        return;
      }
      const now = performance.now();
      const dt = Math.max(1, now - pointerState.lastMoveTime) / 1000;
      const dx = e.offsetX - pointerState.lastX, dy = e.offsetY - pointerState.lastY;
      pointerState.lastX = e.offsetX; pointerState.lastY = e.offsetY;
      pointerState.lastMoveTime = now; pointerState.moved += Math.abs(dx) + Math.abs(dy);
      if (pointerState.mode === 'pan') {
        const ps = camera.distance * 0.0014;
        camera.target[0] -= Math.cos(camera.yaw) * dx * ps;
        camera.target[2] += Math.sin(camera.yaw) * dx * ps;
        camera.target[1] += dy * ps;
      } else {
        const dYaw = -dx * 0.0052, dPitch = dy * 0.0052;
        camera.yaw += dYaw; camera.pitch = clamp(camera.pitch + dPitch, -1.45, 1.45);
        camera.velYaw = clamp(lerp(camera.velYaw, dYaw / dt, 0.3), -1.4, 1.4);
        camera.velPitch = clamp(lerp(camera.velPitch, dPitch / dt, 0.3), -1.4, 1.4);
      }
      state.needsRender = true;
    });

    const end = (e) => {
      pointerState.pointers.delete(e.pointerId);
      try { stage.releasePointerCapture(e.pointerId); } catch (_) {}
      if (pointerState.mode === 'pinch') {
        if (pointerState.pointers.size < 2) { pointerState.dragging = false; pointerState.mode = 'orbit'; }
        return;
      }
      if (!pointerState.dragging) return;
      pointerState.dragging = false; stage.classList.remove('grabbing');
      if (performance.now() - pointerState.lastMoveTime > 90) { camera.velYaw = camera.velPitch = 0; }
      // click behavior: a ghost marker or highlighted neighbor centers the
      // camera (keeps the selection); empty space dismisses the comparison.
      if (pointerState.moved < 5 && pointerState.mode === 'orbit' && e.type === 'pointerup') {
        const ghost = pickGhostAt(e.offsetX, e.offsetY);
        if (ghost) { centerCameraOn(ghost); return; }
        const picked = pickPointAt(e.offsetX, e.offsetY);
        if (picked >= 0) {
          const pos = state.positions;
          centerCameraOn([pos[picked * 3], pos[picked * 3 + 1], pos[picked * 3 + 2]]);
        } else if (hasSelection()) {
          state.slots.forEach((s) => { s.text = ''; });
          renderSlots();
          dismissSelection();
        }
      }
    };
    stage.addEventListener('pointerup', end);
    stage.addEventListener('pointercancel', end);
    stage.addEventListener('pointerleave', () => { if (!pointerState.dragging) { state.hoverIndex = -1; stage.classList.remove('point-hover'); } });
    stage.addEventListener('wheel', (e) => {
      if (!overCanvas(e)) return;
      e.preventDefault(); camera.tween = null;
      // zoom keeps following (distance is independent of target) but a manual
      // orbit/pan already cleared follow via pointerdown
      camera.distanceTarget = clamp(camera.distanceTarget * Math.exp(e.deltaY * 0.0011), 0.2, 120);
    }, { passive: false });
  }

  function bindUi() {
    dom.runButton.addEventListener('click', runAll);
    dom.addSlot.addEventListener('click', () => addSlot());
    dom.resetButton.addEventListener('click', resetCamera);
    dom.clearButton.addEventListener('click', () => { state.slots.forEach((s) => { s.text = ''; }); renderSlots(); dismissSelection(); });

    dom.topNInput.addEventListener('input', () => {
      state.topN = Number(dom.topNInput.value); dom.topNValue.textContent = String(state.topN);
    });
    dom.sizeInput.addEventListener('input', () => {
      state.pointScale = Number(dom.sizeInput.value) / 100;
      dom.sizeValue.textContent = `${state.pointScale.toFixed(1)}×`; state.needsRender = true;
    });
    dom.fadeInput.addEventListener('input', () => {
      state.depthFade = Number(dom.fadeInput.value) / 100;
      dom.fadeValue.textContent = `${dom.fadeInput.value}%`; state.needsRender = true;
    });
    dom.dimInput.addEventListener('input', () => {
      state.dimming = Number(dom.dimInput.value) / 100;
      dom.dimValue.textContent = `${dom.dimInput.value}%`; state.needsRender = true;
    });
    dom.colorModeToggle.addEventListener('change', () => {
      state.sourceColorNodes = dom.colorModeToggle.checked; state.needsRender = true;
    });
    dom.morphSlider.addEventListener('input', () => {
      if (state.playing) stopAutoplay(); // manual drag takes over
      morphTo(Number(dom.morphSlider.value) / 100);
    });
    dom.followToggle.addEventListener('change', () => {
      state.followResult = dom.followToggle.checked;
      if (!state.followResult) camera.follow = null;
      else if (state.display[0] && state.display[0].ghost) camera.follow = state.display[0].ghost.slice();
    });
    dom.precomputeToggle.addEventListener('change', () => {
      state.precise = dom.precomputeToggle.checked;
      if (state.precise && state.morph && !state.morph.cache) precomputeMorph();
      else if (state.morph) updateMorphLabel(Number(dom.morphSlider.value) / 100);
      refreshMorphControls();
    });
    dom.playButton.addEventListener('click', toggleAutoplay);
    dom.modelSelect.addEventListener('change', () => {
      const model = dom.modelSelect.value;
      loadTier(state.tierSize, model, pickPrecision(state.tierSize, model));
    });
    dom.datasetSelect.addEventListener('change', () => {
      const size = Number(dom.datasetSelect.value);
      const model = precisionsFor(size, state.model).length ? state.model : Object.keys(tierEntry(size).models)[0];
      loadTier(size, model, pickPrecision(size, model));
    });
    dom.precisionSelect.addEventListener('change', () => {
      loadTier(state.tierSize, state.model, dom.precisionSelect.value);
    });

    dom.slots.closest('.controls-panel').addEventListener('click', (e) => {
      const chip = e.target.closest('[data-example]');
      if (!chip) return;
      const pair = EXAMPLES[chip.dataset.example];
      if (!pair) return;
      state.slots = [];
      pair.forEach((t) => addSlot(t));
      runAll();
    });

    window.addEventListener('resize', resize);
    if (typeof ResizeObserver !== 'undefined') new ResizeObserver(resize).observe(dom.stage);
  }

  async function fetchStars() {
    try {
      const r = await fetch('https://api.github.com/repos/Allexsen/embedding3d');
      if (!r.ok) return;
      const d = await r.json();
      if (typeof d.stargazers_count !== 'number') return;
      const n = d.stargazers_count;
      const fmt = n < 1000 ? String(n) : n < 999500 ? `${(Math.round(n / 100) / 10).toString().replace(/\.0$/, '')}K` : `${(Math.round(n / 1e5) / 10).toString().replace(/\.0$/, '')}M`;
      dom.starCount.textContent = `★ ${fmt}`; dom.starCount.hidden = false;
    } catch (_) {}
  }

  async function start() {
    renderer.init();
    resize();
    bindStage();
    bindUi();
    addSlot();
    addSlot();
    fetchStars();

    try {
      index = await (await fetch('data/index.json')).json();
      state.model = index.defaultModel;
      state.precision = index.defaultPrecision;
      state.tierSize = index.default;
      const m = index.models.find((x) => x.key === state.model);
      state.browserId = m ? m.browserId : state.browserId;
      populateSelectors();
      await loadTier(index.default, index.defaultModel, index.defaultPrecision);
    } catch (err) {
      setStatus('Could not load corpus index. Serve over HTTP.');
      console.error(err);
    }

    requestAnimationFrame((now) => { lastFrameTime = now; requestAnimationFrame(frame); });
  }

  start();
}());

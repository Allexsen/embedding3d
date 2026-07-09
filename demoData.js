(function () {
  'use strict';

  const VECTOR_SIZE = 50;
  const PROJECTION_SIZE = 3;

  // Named axes mapped to vector slots. Slots 44-49 hold per-word hash noise.
  const AXIS = {
    royalty: 0, gender: 1, family: 2, age: 3, animal: 4, tech: 5, vehicle: 6,
    food: 7, companion: 8, wild: 9, portable: 10, sweet: 11, bitter: 12,
    speed: 13, logic: 14, machine: 15, travel: 16, comfort: 17, nature: 18,
    water: 19, sky: 20, valence: 21, intensity: 22, warm: 23, profession: 24,
    care: 25, art: 26, science: 27, size: 28, power: 29, drink: 30,
    place: 31, urban: 32, fr: 33, it: 34, de: 35, jp: 36, uk: 37,
  };

  const GROUPS = {
    royal:     { center: [2.4, 1.1, 1.3],    color: '#ffd56a' },
    family:    { center: [1.6, 1.7, 0.3],    color: '#ff8fb2' },
    animal:    { center: [-2.1, 1.0, -0.6],  color: '#8de28f' },
    tech:      { center: [2.0, -1.2, -1.3],  color: '#77c7ff' },
    transport: { center: [-1.6, -1.3, 0.9],  color: '#a7b0ff' },
    food:      { center: [0.1, -2.0, 0.4],   color: '#ffae67' },
    nature:    { center: [-2.4, -0.2, 1.7],  color: '#7be3c0' },
    emotion:   { center: [0.5, 2.2, -1.2],   color: '#c9a4ff' },
    profession:{ center: [2.8, 0.4, -0.2],   color: '#ff9d7e' },
    place:     { center: [-0.4, 1.1, 2.3],   color: '#9db8d9' },
    attribute: { center: [0.3, 0.3, -2.3],   color: '#9aa7bd' },
  };

  const E = (word, group, axes) => ({ word, group, axes });

  const ENTRIES = [
    // --- royal
    E('king', 'royal', { royalty: 1, gender: 0.9, age: 0.4, power: 0.7 }),
    E('queen', 'royal', { royalty: 1, gender: -0.9, age: 0.4, power: 0.6 }),
    E('prince', 'royal', { royalty: 0.9, gender: 0.8, age: -0.7, power: 0.4 }),
    E('princess', 'royal', { royalty: 0.9, gender: -0.8, age: -0.7, power: 0.35 }),
    E('emperor', 'royal', { royalty: 1, gender: 0.9, age: 0.5, power: 0.9 }),
    E('empress', 'royal', { royalty: 1, gender: -0.9, age: 0.5, power: 0.85 }),
    E('crown', 'royal', { royalty: 0.95, power: 0.5, portable: 0.3 }),
    E('throne', 'royal', { royalty: 0.95, power: 0.6, comfort: 0.2 }),
    E('castle', 'royal', { royalty: 0.8, place: 0.4, size: 0.8 }),
    E('knight', 'royal', { royalty: 0.6, gender: 0.7, power: 0.6, travel: 0.3 }),

    // --- family / people
    E('man', 'family', { gender: 1, age: 0.3 }),
    E('woman', 'family', { gender: -1, age: 0.3 }),
    E('boy', 'family', { gender: 0.9, age: -1 }),
    E('girl', 'family', { gender: -0.9, age: -1 }),
    E('father', 'family', { gender: 0.9, family: 1, age: 0.5, care: 0.4 }),
    E('mother', 'family', { gender: -0.9, family: 1, age: 0.5, care: 0.6 }),
    E('brother', 'family', { gender: 0.9, family: 0.9, age: -0.2 }),
    E('sister', 'family', { gender: -0.9, family: 0.9, age: -0.2 }),
    E('son', 'family', { gender: 0.9, family: 0.9, age: -0.8 }),
    E('daughter', 'family', { gender: -0.9, family: 0.9, age: -0.8 }),
    E('uncle', 'family', { gender: 0.8, family: 0.8, age: 0.4 }),
    E('aunt', 'family', { gender: -0.8, family: 0.8, age: 0.4 }),
    E('grandfather', 'family', { gender: 0.8, family: 1, age: 1 }),
    E('grandmother', 'family', { gender: -0.8, family: 1, age: 1 }),
    E('husband', 'family', { gender: 0.9, family: 1, age: 0.4, valence: 0.3 }),
    E('wife', 'family', { gender: -0.9, family: 1, age: 0.4, valence: 0.3 }),
    E('baby', 'family', { age: -1.2, family: 0.7, care: 0.8, size: -0.8 }),
    E('child', 'family', { age: -1, family: 0.6, size: -0.5 }),
    E('friend', 'family', { companion: 0.9, valence: 0.5 }),
    E('person', 'family', { family: 0.2, gender: 0.05 }),

    // --- animals
    E('dog', 'animal', { animal: 1, companion: 0.9, comfort: 0.4 }),
    E('cat', 'animal', { animal: 1, companion: 0.7, comfort: 0.4 }),
    E('puppy', 'animal', { animal: 1, companion: 0.9, age: -1, size: -0.6 }),
    E('kitten', 'animal', { animal: 1, companion: 0.7, age: -1, size: -0.6 }),
    E('wolf', 'animal', { animal: 1, wild: 0.95, power: 0.5 }),
    E('fox', 'animal', { animal: 1, wild: 0.6, logic: 0.4 }),
    E('bird', 'animal', { animal: 0.9, sky: 0.8, size: -0.4 }),
    E('eagle', 'animal', { animal: 0.9, sky: 0.9, wild: 0.7, power: 0.5 }),
    E('horse', 'animal', { animal: 1, speed: 0.6, travel: 0.5, size: 0.5 }),
    E('lion', 'animal', { animal: 1, wild: 0.9, power: 0.9, size: 0.6 }),
    E('tiger', 'animal', { animal: 1, wild: 0.9, power: 0.8, size: 0.6 }),
    E('bear', 'animal', { animal: 1, wild: 0.8, power: 0.8, size: 0.7 }),
    E('rabbit', 'animal', { animal: 1, size: -0.5, speed: 0.4 }),
    E('mouse', 'animal', { animal: 1, size: -0.9 }),
    E('fish', 'animal', { animal: 0.9, water: 0.9 }),
    E('shark', 'animal', { animal: 0.9, water: 0.9, wild: 0.8, power: 0.7 }),
    E('whale', 'animal', { animal: 0.9, water: 0.9, size: 1 }),
    E('cow', 'animal', { animal: 1, food: 0.3, size: 0.6 }),
    E('sheep', 'animal', { animal: 1, size: 0.3, comfort: 0.3 }),
    E('pig', 'animal', { animal: 1, food: 0.4, size: 0.4 }),

    // --- tech
    E('code', 'tech', { tech: 1, logic: 1 }),
    E('data', 'tech', { tech: 1, logic: 0.8, science: 0.4 }),
    E('bug', 'tech', { tech: 0.9, logic: 0.5 }),
    E('laptop', 'tech', { tech: 1, portable: 1, machine: 0.5 }),
    E('phone', 'tech', { tech: 1, portable: 0.9, comfort: 0.3, machine: 0.4 }),
    E('computer', 'tech', { tech: 1, machine: 0.8, logic: 0.7 }),
    E('robot', 'tech', { tech: 1, machine: 1, power: 0.3 }),
    E('software', 'tech', { tech: 1, logic: 0.8 }),
    E('internet', 'tech', { tech: 1, logic: 0.5, science: 0.3 }),
    E('network', 'tech', { tech: 0.9, logic: 0.5 }),
    E('server', 'tech', { tech: 0.95, machine: 0.6 }),
    E('screen', 'tech', { tech: 0.8, portable: 0.4 }),
    E('keyboard', 'tech', { tech: 0.8, portable: 0.5 }),
    E('algorithm', 'tech', { tech: 0.9, logic: 1, science: 0.5 }),
    E('ai', 'tech', { tech: 1, logic: 0.9, science: 0.6 }),
    E('email', 'tech', { tech: 0.8, logic: 0.3 }),
    E('web', 'tech', { tech: 0.9, logic: 0.4 }),
    E('digital', 'tech', { tech: 0.9, logic: 0.3 }),
    E('science', 'tech', { science: 1, logic: 0.7 }),

    // --- transport
    E('car', 'transport', { vehicle: 1, speed: 0.8, machine: 0.5 }),
    E('truck', 'transport', { vehicle: 1, speed: 0.4, machine: 0.6, size: 0.6 }),
    E('train', 'transport', { vehicle: 1, travel: 0.9, machine: 0.6, size: 0.7 }),
    E('bike', 'transport', { vehicle: 1, speed: 0.5, portable: 0.4 }),
    E('bus', 'transport', { vehicle: 1, travel: 0.7, size: 0.6 }),
    E('plane', 'transport', { vehicle: 1, sky: 0.9, speed: 0.95, travel: 0.9 }),
    E('boat', 'transport', { vehicle: 0.9, water: 0.9, travel: 0.6 }),
    E('ship', 'transport', { vehicle: 0.9, water: 0.9, travel: 0.8, size: 0.8 }),
    E('rocket', 'transport', { vehicle: 0.9, sky: 1, speed: 1, science: 0.5 }),
    E('engine', 'transport', { vehicle: 0.6, machine: 0.9, power: 0.6 }),
    E('road', 'transport', { vehicle: 0.5, travel: 0.7, place: 0.3 }),

    // --- food & drink
    E('apple', 'food', { food: 1, sweet: 0.8, nature: 0.3 }),
    E('orange', 'food', { food: 1, sweet: 0.7, nature: 0.3 }),
    E('banana', 'food', { food: 1, sweet: 0.8, nature: 0.3 }),
    E('lemon', 'food', { food: 1, bitter: 0.5, nature: 0.3 }),
    E('bread', 'food', { food: 1, comfort: 0.7 }),
    E('cheese', 'food', { food: 1, comfort: 0.5 }),
    E('pizza', 'food', { food: 1, comfort: 0.7, warm: 0.5 }),
    E('cake', 'food', { food: 1, sweet: 1, comfort: 0.6 }),
    E('chocolate', 'food', { food: 1, sweet: 1, comfort: 0.5 }),
    E('sugar', 'food', { food: 0.8, sweet: 1 }),
    E('salt', 'food', { food: 0.8, bitter: 0.3 }),
    E('honey', 'food', { food: 0.9, sweet: 0.95, nature: 0.4 }),
    E('rice', 'food', { food: 1, comfort: 0.4 }),
    E('soup', 'food', { food: 1, warm: 0.7, comfort: 0.6 }),
    E('meat', 'food', { food: 1, animal: 0.3 }),
    E('fruit', 'food', { food: 1, sweet: 0.7, nature: 0.5 }),
    E('coffee', 'food', { food: 0.7, drink: 1, bitter: 0.8, warm: 0.6 }),
    E('tea', 'food', { food: 0.7, drink: 1, comfort: 0.7, warm: 0.6 }),
    E('milk', 'food', { food: 0.8, drink: 0.9, comfort: 0.5 }),
    E('wine', 'food', { food: 0.6, drink: 1, bitter: 0.4 }),
    E('beer', 'food', { food: 0.6, drink: 1, bitter: 0.5 }),

    // --- nature
    E('tree', 'nature', { nature: 1, size: 0.6 }),
    E('flower', 'nature', { nature: 1, art: 0.3, size: -0.4 }),
    E('forest', 'nature', { nature: 1, size: 0.8, wild: 0.5 }),
    E('grass', 'nature', { nature: 0.9, size: -0.6 }),
    E('river', 'nature', { nature: 0.9, water: 1, travel: 0.3 }),
    E('lake', 'nature', { nature: 0.9, water: 0.95 }),
    E('ocean', 'nature', { nature: 0.9, water: 1, size: 1 }),
    E('sea', 'nature', { nature: 0.9, water: 1, size: 0.9 }),
    E('mountain', 'nature', { nature: 1, size: 1, place: 0.4 }),
    E('stone', 'nature', { nature: 0.8, size: 0.3, power: 0.3 }),
    E('sky', 'nature', { nature: 0.8, sky: 1 }),
    E('sun', 'nature', { nature: 0.8, sky: 0.9, warm: 1, power: 0.6 }),
    E('moon', 'nature', { nature: 0.8, sky: 1, warm: -0.4 }),
    E('star', 'nature', { nature: 0.8, sky: 1, science: 0.3 }),
    E('cloud', 'nature', { nature: 0.8, sky: 0.9, water: 0.4 }),
    E('rain', 'nature', { nature: 0.9, water: 0.8, sky: 0.7 }),
    E('snow', 'nature', { nature: 0.9, water: 0.6, sky: 0.6, warm: -0.9 }),
    E('ice', 'nature', { nature: 0.7, water: 0.7, warm: -1 }),
    E('wind', 'nature', { nature: 0.9, sky: 0.6, speed: 0.5 }),
    E('storm', 'nature', { nature: 0.9, sky: 0.7, intensity: 0.9, power: 0.7 }),
    E('fire', 'nature', { nature: 0.6, warm: 1, intensity: 0.9, power: 0.6 }),
    E('earth', 'nature', { nature: 0.9, place: 0.5, size: 0.9 }),
    E('water', 'nature', { nature: 0.7, water: 1, drink: 0.7 }),

    // --- emotion
    E('happy', 'emotion', { valence: 1, intensity: 0.5 }),
    E('joy', 'emotion', { valence: 1, intensity: 0.7 }),
    E('love', 'emotion', { valence: 0.9, intensity: 0.9, care: 0.6 }),
    E('hope', 'emotion', { valence: 0.8, intensity: 0.4 }),
    E('calm', 'emotion', { valence: 0.5, intensity: -0.8, comfort: 0.6 }),
    E('smile', 'emotion', { valence: 0.9, intensity: 0.4 }),
    E('dream', 'emotion', { valence: 0.5, sky: 0.3, art: 0.4 }),
    E('sad', 'emotion', { valence: -1, intensity: 0.5 }),
    E('tears', 'emotion', { valence: -0.8, intensity: 0.6, water: 0.3 }),
    E('angry', 'emotion', { valence: -0.9, intensity: 0.9 }),
    E('fear', 'emotion', { valence: -0.9, intensity: 0.8 }),
    E('hate', 'emotion', { valence: -1, intensity: 0.9 }),

    // --- professions
    E('doctor', 'profession', { profession: 1, care: 0.9, science: 0.5 }),
    E('nurse', 'profession', { profession: 1, care: 1, gender: -0.3 }),
    E('teacher', 'profession', { profession: 1, care: 0.6, logic: 0.4 }),
    E('student', 'profession', { profession: 0.7, age: -0.6, logic: 0.4 }),
    E('engineer', 'profession', { profession: 1, tech: 0.6, logic: 0.7, machine: 0.4 }),
    E('scientist', 'profession', { profession: 1, science: 1, logic: 0.8 }),
    E('artist', 'profession', { profession: 1, art: 1 }),
    E('writer', 'profession', { profession: 1, art: 0.7, logic: 0.3 }),
    E('singer', 'profession', { profession: 1, art: 0.8, intensity: 0.4 }),
    E('chef', 'profession', { profession: 1, food: 0.7 }),
    E('farmer', 'profession', { profession: 1, nature: 0.6, food: 0.5 }),
    E('soldier', 'profession', { profession: 1, power: 0.7, intensity: 0.5 }),
    E('police', 'profession', { profession: 1, power: 0.6, care: 0.3 }),
    E('lawyer', 'profession', { profession: 1, logic: 0.7, power: 0.4 }),
    E('pilot', 'profession', { profession: 1, sky: 0.8, vehicle: 0.6, speed: 0.6 }),

    // --- places (country/city pairs share a nation axis, so
    //     paris - france + italy lands on rome)
    E('france', 'place', { place: 1, urban: -1, fr: 1 }),
    E('paris', 'place', { place: 1, urban: 1, fr: 1, art: 0.4 }),
    E('italy', 'place', { place: 1, urban: -1, it: 1 }),
    E('rome', 'place', { place: 1, urban: 1, it: 1, age: 0.5 }),
    E('germany', 'place', { place: 1, urban: -1, de: 1 }),
    E('berlin', 'place', { place: 1, urban: 1, de: 1 }),
    E('japan', 'place', { place: 1, urban: -1, jp: 1, tech: 0.3 }),
    E('tokyo', 'place', { place: 1, urban: 1, jp: 1, tech: 0.4 }),
    E('england', 'place', { place: 1, urban: -1, uk: 1 }),
    E('london', 'place', { place: 1, urban: 1, uk: 1 }),
    E('city', 'place', { place: 0.8, urban: 1, size: 0.7 }),
    E('country', 'place', { place: 0.8, urban: -1, size: 0.8 }),
    E('village', 'place', { place: 0.8, urban: 0.5, size: -0.6, nature: 0.4 }),
    E('home', 'place', { place: 0.6, comfort: 0.9, family: 0.5 }),
    E('school', 'place', { place: 0.6, logic: 0.5, age: -0.5, care: 0.4 }),
    E('hospital', 'place', { place: 0.6, care: 0.9, science: 0.4 }),

    // --- attribute words (useful in arithmetic)
    E('big', 'attribute', { size: 1 }),
    E('small', 'attribute', { size: -1 }),
    E('fast', 'attribute', { speed: 1 }),
    E('slow', 'attribute', { speed: -1 }),
    E('hot', 'attribute', { warm: 1 }),
    E('cold', 'attribute', { warm: -1 }),
    E('strong', 'attribute', { power: 1 }),
    E('weak', 'attribute', { power: -1 }),
    E('old', 'attribute', { age: 1 }),
    E('young', 'attribute', { age: -1 }),
    E('good', 'attribute', { valence: 0.8 }),
    E('bad', 'attribute', { valence: -0.8 }),
  ];

  function hashWord(word, seed) {
    let hash = seed >>> 0;
    for (let i = 0; i < word.length; i++) {
      hash = (hash * 31 + word.charCodeAt(i)) >>> 0;
      hash ^= hash >>> 13;
      hash = (hash * 2654435761) >>> 0;
    }
    return hash;
  }

  function unitNoise(word, seed) {
    return (hashWord(word, seed) & 1023) / 1023 - 0.5;
  }

  function buildVector(entry) {
    const vector = new Float32Array(VECTOR_SIZE);
    for (const [axis, value] of Object.entries(entry.axes)) {
      const index = AXIS[axis];
      if (index !== undefined) {
        vector[index] = value;
      }
    }
    for (let i = 0; i < 6; i++) {
      vector[44 + i] = unitNoise(entry.word, 97 + i * 41) * 0.04;
    }
    return vector;
  }

  function buildDataset() {
    const words = [];
    const embeddings = new Float32Array(ENTRIES.length * VECTOR_SIZE);
    const projected = new Float32Array(ENTRIES.length * PROJECTION_SIZE);
    const baseColors = [];

    for (let i = 0; i < ENTRIES.length; i++) {
      const entry = ENTRIES[i];
      const group = GROUPS[entry.group];
      const vector = buildVector(entry);

      words.push(entry.word);
      embeddings.set(vector, i * VECTOR_SIZE);
      projected[i * 3] = group.center[0] + unitNoise(entry.word, 3) * 1.15;
      projected[i * 3 + 1] = group.center[1] + unitNoise(entry.word, 7) * 1.15;
      projected[i * 3 + 2] = group.center[2] + unitNoise(entry.word, 11) * 1.15;
      baseColors.push(group.color);
    }

    return {
      words,
      embeddings,
      projected,
      baseColors,
      vectorSize: VECTOR_SIZE,
      projectionSize: PROJECTION_SIZE,
      count: ENTRIES.length,
    };
  }

  if (typeof window !== 'undefined') {
    window.EMBEDDING_DEMO = buildDataset();
  } else if (typeof module !== 'undefined') {
    module.exports = buildDataset();
  }
}());

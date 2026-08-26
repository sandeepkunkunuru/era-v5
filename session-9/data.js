/* ERA v5 · Session 9 — every number on the page, in one place.
   Measurements come from era-v5/session-09-loss-harness/results.json (seed 20260822,
   tiny Shakespeare, GPT-2 BPE V=50,257, RTX 4050). Lecture facts carry a [hh:mm:ss]
   stamp verified against our own transcript. */

/* --- V5's shape, which is what the arithmetic below is about --- */
export const V5 = { V: 131072, d_model: 4096, layers: 48, kv_heads: 8, head_dim: 128, bytes: 2 };

/* --- the three bills, at a range of context lengths ---------------------------
   Two of these are memory and one is compute, so they are NOT plotted together;
   the chart shows the two MEMORY bills on one axis, in GiB. Honest caveat, stated
   on the page: the KV cache is an inference cost that persists per conversation,
   the logits tensor is a training-time transient at the last layer. Same units,
   different phases. */
export const CONTEXTS = [1024, 4096, 8192, 32768, 131072, 262144, 1048576];

export const kvBytesPerToken =
  2 * V5.layers * V5.kv_heads * V5.head_dim * V5.bytes;      // 196,608 B/token
export const logitBytesPerToken = V5.V * V5.bytes;           // 262,144 B/token

export const BILLS = CONTEXTS.map(T => ({
  T,
  kv:        kvBytesPerToken * T / 2 ** 30,
  logits:    logitBytesPerToken * T / 2 ** 30,
  logitsBwd: 2 * logitBytesPerToken * T / 2 ** 30,
}));

/* --- requirement 7, measured ------------------------------------------------- */
export const CHUNKING = {
  modelScale: {
    label: "this harness's head", N: 4096, D: 384, V: 50257,
    floor: 366.2, ratio: 5.99,
    rows: [
      { impl: "naive", peak: 2650.5, loss: 10.910684586 },
      { impl: "chunk = 1024", peak: 958.0, loss: 10.910684586 },
      { impl: "chunk = 256", peak: 516.3, loss: 10.910684586 },
      { impl: "chunk = 64", peak: 442.7, loss: 10.910687447 },
    ],
  },
  v5Scale: {
    label: "V5's vocabulary", N: 2048, D: 1024, V: 131072,
    floor: 1026.4, ratio: 2.32,
    rows: [
      { impl: "naive", peak: 3586.1, loss: 12.022954941 },
      { impl: "chunk = 512", peak: 1799.1, loss: 12.022954941 },
      { impl: "chunk = 128", peak: 1543.3, loss: 12.022954941 },
      { impl: "chunk = 32", peak: 1543.2, loss: 12.022954941 },
    ],
  },
};

/* --- the four quiet bugs, measured on a trained model ------------------------ */
export const BUGS = [
  {
    name: "padding counted into the mean",
    what: "A pad token is not a prediction. Include it and the loss reports how good you are at predicting nothing.",
    measured: "Trained on padded batches, the counted loss fell 6.63 nats and the honest loss fell 0.21.",
    headline: "3%", headlineLabel: "of the apparent progress was real",
    stamp: "01:27:54",
  },
  {
    name: "predicting across a document boundary",
    what: "Packing puts unrelated documents end to end. The join asks the model to predict a Python file from a Shakespeare speech.",
    measured: "Boundary positions cost 17.48 nats against 6.55 elsewhere.",
    headline: "2.67×", headlineLabel: "what an ordinary position costs",
    stamp: "01:30:56",
  },
  {
    name: "the wrong denominator",
    what: "Divide by B×T instead of the number of positions that actually counted, and the loss is scaled by your padding ratio — which changes every batch.",
    measured: "Same logits, same targets: 4.2502 correct against 2.5636 with B×T.",
    headline: "−40%", headlineLabel: "purely from the divisor",
    stamp: "01:29:55",
  },
  {
    name: "the shift, the wrong way",
    what: "Forget to shift and the model is handed its own input as the answer. The loss curve is beautiful and the model has learned to copy.",
    measured: "Caught by printing token strings side by side, never by reading ids.",
    headline: "print", headlineLabel: "the strings. every time.",
    stamp: "01:29:55",
  },
];

/* --- Part 2: two heads ------------------------------------------------------- */
export const MTP = {
  lnV: 10.8249,
  tied: [
    { step: 0, l1: 10.8619, l2: 10.8681 }, { step: 10, l1: 9.7080, l2: 9.5722 },
    { step: 20, l1: 8.7211, l2: 8.6034 },  { step: 30, l1: 7.8796, l2: 7.7595 },
    { step: 40, l1: 7.0928, l2: 7.0354 },  { step: 50, l1: 6.7028, l2: 6.7195 },
    { step: 75, l1: 6.1102, l2: 6.2892 },  { step: 100, l1: 6.0837, l2: 6.3317 },
    { step: 150, l1: 5.4774, l2: 5.8052 }, { step: 200, l1: 5.4394, l2: 5.8987 },
    { step: 300, l1: 5.1086, l2: 5.6747 }, { step: 450, l1: 4.7146, l2: 5.3585 },
    { step: 600, l1: 4.3865, l2: 5.1061 }, { step: 750, l1: 4.2940, l2: 5.0000 },
    { step: 900, l1: 4.1171, l2: 4.8527 }, { step: 1050, l1: 4.1013, l2: 4.8080 },
    { step: 1200, l1: 3.9198, l2: 4.6849 },{ step: 1350, l1: 3.7445, l2: 4.4791 },
    { step: 1500, l1: 3.7687, l2: 4.4709 },
  ],
  crossoverSteps: [10, 20, 30, 40],
  unigramEntropy: 6.3151,
  crossoverAt: 6.90,
  hypotheses: [
    {
      name: "tying",
      claim: "Head 1 shares its matrix with the input embedding, so its gradient is doing two jobs; head 2 is a free matrix. Untie head 1 and the crossover should vanish.",
      test: "Identical seed, data and steps, tie_weights = False.",
      result: "refuted",
      because: "The untied arm crosses over at the same steps — 0, 10, 20, 30, 40 — and by the same margin. Slightly earlier, if anything.",
    },
    {
      name: "the marginal-fitting phase",
      claim: "Before the trunk carries usable context, t+1 has no advantage over t+2. So the crossover should sit where the model stops fitting unigram frequencies.",
      test: "Compute the corpus unigram entropy and check whether the crossover window contains it.",
      result: "refuted",
      because: "Unigram entropy is 6.32 nats; the crossover happens at about 6.90 — roughly 0.6 nats above the marginal floor, before the model has finished fitting the marginal at all.",
    },
  ],
};

/* --- corrections worth carrying forward -------------------------------------- */
export const CORRECTIONS = [
  { stamp: "00:25:38", said: "Temperature 1 picks the top token; lowering it gives the model more options.",
    fact: "Backwards. Dividing logits by T flattens as T rises and sharpens as T falls: T→0 is greedy, T→∞ is uniform.",
    weight: "repeat", note: "Second session running — S8 taught temperature as top-k renormalisation." },
  { stamp: "01:29:55", said: "Mask &lt;pad&gt; and &lt;eos&gt; alike — neither should be part of the learning.",
    fact: "&lt;pad&gt; is a batching artefact and must go. &lt;eos&gt; is a real token the model has to learn to emit; mask it and the model never learns to stop.",
    weight: "code", note: "Opposite decisions about superficially similar tokens." },
  { stamp: "02:31:03", said: "The heads physically split the residual stream into blocks; head mixing happens in the FFN.",
    fact: "Each head has its own projection reading the full D-vector, and W_O — a full D×D matrix — mixes every head inside the attention sub-layer, before the FFN.",
    weight: "concept", note: "He counted W_O himself 90 minutes earlier." },
  { stamp: "01:40:59", said: "We took our kernel from ~4,000 tok/s to ~50,000 — a 10× speed-up.",
    fact: "The two ends are different models. Cookbook P1 9B = 3,535 tok/s; P2 2B = 50,858. Held at one model it is 38,179 → 50,858 = +33% — with a better loss, 7.0009 → 6.6337.",
    weight: "v4", note: "The cookbook's framing is sharper than the lecture's." },
  { stamp: "02:18:50", said: "Chinchilla says you can fix things on a small model then train a bigger one.",
    fact: "Chinchilla (2203.15556) is compute-optimal scaling — ~20 tokens per parameter. Fixing hyperparameters on a proxy and transferring is µTransfer (2203.03466).",
    weight: "concept", note: "Both matter to V5; they answer different questions." },
  { stamp: "01:34:58", said: "bfloat16 came from DeepMind.",
    fact: "Google Brain, for TPUs — 'brain float' is named after the team.",
    weight: "repeat", note: "Corrected in the Session 7 report. It did not take." },
  { stamp: "02:06:37", said: "Fill-in-the-middle is not used anymore.",
    fact: "Standard in every current code model, and it is why in-editor completion works inside a function. The written brief says so too.",
    weight: "code", note: "The recording contradicts its own source." },
  { stamp: "02:23:53", said: "OpenAI gives the log probabilities of their vocabulary.",
    fact: "top_logprobs caps at 20 tokens per position — a truncated top-k, not a distribution. Which is exactly why the one-hot fallback is the only option.",
    weight: "concept", note: "Right conclusion, wrong reason." },
];

#![allow(dead_code)]

const N_LAYERS: usize = 96;
const D_MODEL: usize = 128 * N_LAYERS;
const N_MLP: usize = 4 * D_MODEL;
const D_HEAD: usize = 128;
const N_HEADS: usize = D_MODEL / D_HEAD;
const N_VOCAB: usize = 50_000;

type Token = u64;
type Logits = [f32; N_VOCAB];

trait ARModel {
    fn apply(&self, tokens: &[Token]) -> Vec<Logits>;
}

#[derive(Clone)]
struct State([f32; D_MODEL]);

type Query = State;
type Update = State;

impl State {
    fn zero() -> Self {
        State([0.0; D_MODEL])
    }

    fn update(&self, right: &Update) -> State {
        let mut out = self.clone();
        for (i, r) in right.0.iter().enumerate() {
            out.0[i] += r;
        }
        out
    }

    fn query(&self, right: &Query) -> f32 {
        dot(&self.0, &right.0)
    }
}

fn dot<const N: usize>(l: &[f32; N], r: &[f32; N]) -> f32 {
    let mut out = 0.0;
    for (i, r) in r.iter().enumerate() {
        out += l[i] * r;
    }
    out
}

struct Transformer {
    embedding: Embedding,
    layers: [ResBlock; N_LAYERS],
    unembedding: Unembedding,
}

struct Embedding([State; N_VOCAB]);

impl Embedding {
    fn apply(&self, tok: Token) -> State {
        self.0[tok as usize].clone()
    }
}

struct LogitFn(Query);
impl LogitFn {
    fn apply(&self, st: &State) -> f32 {
        self.0.query(st)
    }
}

struct Unembedding([LogitFn; N_VOCAB]);

impl Unembedding {
    fn apply(&self, state: &State) -> Logits {
        let mut out: Logits = [0.0; N_VOCAB];
        for (i, f) in self.0.iter().enumerate() {
            out[i] = f.apply(state);
        }
        out
    }
}

struct ResBlock {
    attn: AttnLayer,
    mlps: MLPLayer,
}

struct AttnLayer {
    heads: [AttnHead; N_HEADS],
}

type AttnVector = [f32; D_HEAD];

struct AttnHead {
    W_Q: Box<dyn Fn(&State) -> AttnVector>,
    W_K: Box<dyn Fn(&State) -> AttnVector>,
    W_V: Box<dyn Fn(&State) -> AttnVector>,
    W_O: Box<dyn Fn(&AttnVector) -> Update>,
}

fn softmax(_scores: &mut [f32]) {
    // ...
}

impl AttnHead {
    fn apply(&self, states: &[State]) -> Vec<Update> {
        // Apply the Q, K, and V projections to produce Q, K, and V
        // vectors for each token position.
        let qs: Vec<AttnVector> = states.iter().map(&self.W_Q).collect();
        let ks: Vec<AttnVector> = states.iter().map(&self.W_K).collect();
        let vs: Vec<AttnVector> = states.iter().map(&self.W_V).collect();

        let mut values: Vec<_> = states.iter().map(|_| [0.0; D_HEAD]).collect();

        // Iterate over each token position to compute the output at
        // that position
        for (src, my_q) in qs.iter().enumerate() {
            // Each position may attend to any earlier position. We
            // compute an attention "score" between the current
            // position and each earlier position by dot-producting
            // our Q vector with their K vector.
            let mut scores = Vec::with_capacity(src);

            let visible_indices = 0..=src;

            for i in visible_indices.clone() {
                scores.push(dot(my_q, &ks[i]));
            }

            // We use a softmax to turn that vector of scores into a
            // probability distribution
            softmax(&mut scores);

            // Now we loop over each visible position again, weight
            // their V vector by their attention weight, and sum them
            // all together.
            for i in visible_indices {
                let score = scores[i];
                let v = vs[i];
                for (j, vj) in v.iter().enumerate() {
                    values[src][j] += vj * score;
                }
            }
        }

        // Now we have a value vector for each position. Use the O
        // projection to project it up to a full State vector
        values.iter().map(&self.W_O).collect()
    }
}

impl AttnLayer {
    fn apply(&self, states: &[State]) -> Vec<Update> {
        let mut updates: Vec<Update> = states.iter().map(|_| State::zero()).collect();

        for h in self.heads.iter() {
            let head_out = h.apply(states);

            updates = updates
                .iter()
                .zip(head_out.iter())
                .map(|(l, r)| l.update(r))
                .collect();
        }

        updates
    }
}

struct Neuron {
    read: Query,
    write: Update,
}

struct MLPLayer {
    mlps: [Neuron; N_MLP],
    nonlinear: fn(f32) -> f32,
}

impl MLPLayer {
    fn apply(&self, state: &State) -> Update {
        let mut out: Update = State::zero();
        for mlp in self.mlps.iter() {
            let pre_act = mlp.read.query(state);
            let post_act = (self.nonlinear)(pre_act);
            let unit_out: Update = State(mlp.write.0.map(|f| f * post_act));
            out = out.update(&unit_out)
        }
        out
    }
}

impl ARModel for Transformer {
    fn apply(&self, tokens: &[Token]) -> Vec<Logits> {
        // Embeddings: convert tokens into initial states
        let mut states = tokens
            .iter()
            .map(|t| self.embedding.apply(*t))
            .collect::<Vec<_>>();

        // Pass the hidden state through each layer in turn
        for layer in self.layers.iter() {
            let attn_out = layer.attn.apply(&states);
            states = states
                .iter()
                .zip(attn_out.iter())
                .map(|(l, r)| l.update(r))
                .collect();

            for i in 0..states.len() {
                let mlp_out = layer.mlps.apply(&states[i]);
                states[i] = states[i].update(&mlp_out);
            }
        }

        // Then apply the unembedding to get out logits
        states.iter().map(|s| self.unembedding.apply(s)).collect()
    }
}

struct AlternateAttnHead {
    W_QK: Box<dyn Fn(&State, &State) -> f32>,
    W_OV: Box<dyn Fn(&State) -> Update>,
}

impl AlternateAttnHead {
    fn apply(&self, states: &[State]) -> Vec<Update> {
        let mut output = states
            .iter()
            .map(|_| State::zero())
            .collect::<Vec<Update>>();

        // Iterate over each token position to compute the output at
        // that position
        for (src, src_state) in states.iter().enumerate() {
            // Each position may attend to any earlier position. We
            // compute an attention "score" between the current
            // position and each earlier position by applying the QK circuit to both positions
            let mut scores = Vec::with_capacity(src);

            let visible_indices = 0..=src;

            for i in visible_indices.clone() {
                scores.push((self.W_QK)(&src_state, &states[i]));
            }

            softmax(&mut scores);

            // Now we loop over each visible position again, compute
            // the output update using the OV weight, scale it by the
            // attention weight, and accumulate.
            for i in visible_indices {
                let score = scores[i];
                let o = (self.W_OV)(&states[i]);
                let scaled = State(o.0.map(|f| f * score));
                output[src] = output[src].update(&scaled);
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {}

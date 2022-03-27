#![allow(dead_code)]

const N_LAYERS: usize = 96;
const D_MODEL: usize = 128 * N_LAYERS;
const D_MLP: usize = 4 * D_MODEL;
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

type Query = State;
type Update = State;

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
        let qs: Vec<AttnVector> = states.iter().map(&self.W_Q).collect();
        let ks: Vec<AttnVector> = states.iter().map(&self.W_K).collect();
        let vs: Vec<AttnVector> = states.iter().map(&self.W_V).collect();

        let mut values: Vec<_> = states.iter().map(|_| [0.0; D_HEAD]).collect();

        for (src, my_q) in qs.iter().enumerate() {
            let mut scores = Vec::with_capacity(src);
            for k in ks[0..=src].iter() {
                scores.push(dot(my_q, k));
            }

            softmax(&mut scores);

            for (score, val) in scores.iter().zip(vs.iter()) {
                for (i, v) in val.iter().enumerate() {
                    values[src][i] += v * score;
                }
            }
        }

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
    mlps: [Neuron; D_MLP],
    nonlinear: fn(f32) -> f32,
}

impl MLPLayer {
    fn apply(&self, state: &State) -> Update {
        let mut out = State::zero();
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
        let mut states = tokens
            .iter()
            .map(|t| self.embedding.apply(*t))
            .collect::<Vec<_>>();

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

        states.iter().map(|s| self.unembedding.apply(s)).collect()
    }
}

#[cfg(test)]
mod tests {}

// qwen3-layer-dump.cpp
//
// A narrowed form of llama.cpp's examples/eval-callback. It captures the residual stream output of chosen transformer blocks (default: 9, 18, 27 in HF output_hidden_states numbering) for every token of a prompt, concatenates them along the feature axis, and writes a raw float32 file that service.py turns into the (1, seq_len, 12288) tensor Flux2KleinInpaintPipeline expects, using only the GGUF model through llama.cpp. No safetensors, no PyTorch, no GPU required.
//
// llama.cpp's embedding API only exposes the final layer. An intermediate layer's activations are reachable through the cb_eval hook in llama_context_params, invoked for every tensor in the compute graph.
//
// HF's output_hidden_states has num_layers+1 entries: index 0 is the embedding output, index k the output after the k-th block. llama.cpp's layer loop is 0-indexed, so hidden_states[k] is il = k-1; layers (9, 18, 27) are il = (8, 17, 26).
//
// The end-of-block residual stream (self-attn + FFN residual add, after the control-vector hook: exactly what HF calls hidden_states[il+1]) is named "l_out-<il>" by llama_context::graph_get_cb(). Earlier same-layer tensors such as "ffn_out-<il>" are a different point in the block and are deliberately not matched. llama.cpp's model code changes over time: --dump-tensor-names shows what a given build and GGUF call them.
//
// USAGE
//   llama-qwen3-layer-dump --model M.gguf --prompt-file P.txt --layers 9,18,27 --out raw_layers.bin [--max-tokens N] [--n-ctx N] [--batch-size N]
//   llama-qwen3-layer-dump --model M.gguf --dump-tensor-names

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct CaptureConfig {
    std::vector<int> target_il;       // llama.cpp 0-indexed layer indices to capture
    bool dump_names_only = false;
    std::set<std::string> seen_names; // for --dump-tensor-names
};

struct CaptureState {
    CaptureConfig cfg;
    int n_embd = 0;
    std::vector<std::vector<float>> per_layer_buffers;  // one flat buffer per layer, appended per micro-batch
    std::vector<int> il_order;                          // per_layer_buffers index -> il
};

static int layer_index_from_name(const std::string &name) {
    auto pos = name.find_last_of('-');
    if (pos == std::string::npos) return -1;
    const std::string tail = name.substr(pos + 1);
    if (tail.empty()) return -1;
    for (char c : tail) if (!isdigit((unsigned char)c)) return -1;
    return std::stoi(tail);
}

static const char *kTargetPrefix = "l_out-";

static bool is_target_layer_tensor(const std::string &name, int &il_out) {
    if (name.rfind(kTargetPrefix, 0) != 0) return false;
    il_out = layer_index_from_name(name);
    return il_out >= 0;
}

static bool capture_callback(struct ggml_tensor *t, bool ask, void *user_data) {
    auto *st = static_cast<CaptureState *>(user_data);
    const std::string name = ggml_get_name(t);
    if (st->cfg.dump_names_only) {
        if (ask) {
            if (st->cfg.seen_names.insert(name).second) fprintf(stdout, "%s\n", name.c_str());
            return false;
        }
        return true;
    }
    int il = -1;
    bool wanted = is_target_layer_tensor(name, il) && std::find(st->cfg.target_il.begin(), st->cfg.target_il.end(), il) != st->cfg.target_il.end();
    if (ask) return wanted;
    if (!wanted) return true;

    // The tensor is computed: copy it out as float32 whatever the backend stores. ne1 is this micro-batch's tokens, not the whole sequence.
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1];
    if (st->n_embd == 0) st->n_embd = (int) ne0;
    std::vector<float> host(ne0 * ne1);
    if (t->type == GGML_TYPE_F32) ggml_backend_tensor_get(t, host.data(), 0, ne0 * ne1 * sizeof(float));
    else {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        ggml_get_type_traits(t->type)->to_float(raw.data(), host.data(), ne0 * ne1);
    }
    auto it = std::find(st->il_order.begin(), st->il_order.end(), il);
    if (it == st->il_order.end()) {
        st->il_order.push_back(il);
        st->per_layer_buffers.push_back(host);
    } else {
        auto &existing = st->per_layer_buffers[it - st->il_order.begin()];
        existing.insert(existing.end(), host.begin(), host.end());
    }
    return true;
}

static std::vector<int> parse_layers(const std::string &csv) {
    std::vector<int> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::stoi(item));
    return out;
}

int main(int argc, char **argv) {
    std::string model_path, prompt_file, out_path;
    std::vector<int> hf_layers = {9, 18, 27};
    bool dump_names = false;
    int max_tokens = -1, n_ctx = 8192, n_batch = 2048;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--prompt-file" && i + 1 < argc) prompt_file = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--layers" && i + 1 < argc) hf_layers = parse_layers(argv[++i]);
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::stoi(argv[++i]);
        else if (a == "--n-ctx" && i + 1 < argc) n_ctx = std::stoi(argv[++i]);
        else if (a == "--batch-size" && i + 1 < argc) n_batch = std::stoi(argv[++i]);
        else if (a == "--dump-tensor-names") dump_names = true;
    }
    if (model_path.empty() || (!dump_names && (prompt_file.empty() || out_path.empty()))) {
        fprintf(stderr, "usage: %s --model M.gguf (--dump-tensor-names | --prompt-file F --out O.bin [--layers 9,18,27] [--max-tokens N] [--n-ctx N] [--batch-size N])\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model: %s\n", model_path.c_str()); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);

    CaptureState capture;
    capture.cfg.dump_names_only = dump_names;
    for (int hf_idx : hf_layers) capture.cfg.target_il.push_back(hf_idx - 1);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_batch;
    cparams.cb_eval = capture_callback;
    cparams.cb_eval_user_data = &capture;
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    if (dump_names) {   // one token is enough to walk the whole graph once
        std::vector<llama_token> tok(1);
        int n = llama_tokenize(vocab, " ", 1, tok.data(), (int) tok.size(), true, true);
        if (n < 1) { tok[0] = 0; n = 1; }
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; i++) { batch.token[i] = tok[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = (i == n - 1); }
        batch.n_tokens = n;
        llama_decode(ctx, batch);
        llama_batch_free(batch);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    }

    std::ifstream pf(prompt_file);
    std::stringstream buf; buf << pf.rdbuf();
    std::string prompt_text = buf.str();

    std::vector<llama_token> tokens(prompt_text.size() + 16);
    int n_tokens = llama_tokenize(vocab, prompt_text.c_str(), (int) prompt_text.size(), tokens.data(), (int) tokens.size(), false, true);
    if (n_tokens < 0) { tokens.resize(-n_tokens); n_tokens = llama_tokenize(vocab, prompt_text.c_str(), (int) prompt_text.size(), tokens.data(), (int) tokens.size(), false, true); }
    tokens.resize(n_tokens);

    bool was_truncated = false;
    if (max_tokens > 0 && n_tokens > max_tokens) { was_truncated = true; tokens.resize(max_tokens); n_tokens = max_tokens; }
    fprintf(stderr, "tokenized prompt: %d tokens%s\n", n_tokens, was_truncated ? " (TRUNCATED)" : "");

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) { batch.token[i] = tokens[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = (i == n_tokens - 1); }
    batch.n_tokens = n_tokens;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "llama_decode failed\n"); return 1; }

    if (capture.per_layer_buffers.size() != capture.cfg.target_il.size()) {
        fprintf(stderr, "WARNING: captured %zu layers but %zu were requested: the tensor-name filter does not match this model. Re-run with --dump-tensor-names.\n", capture.per_layer_buffers.size(), capture.cfg.target_il.size());
        return 2;
    }

    // In the order hf_layers was given, which is the order the pipeline concatenates: [n_tokens, n_layers * n_embd] row-major float32.
    std::ofstream out(out_path, std::ios::binary);
    int32_t header[3] = {n_tokens, (int32_t) hf_layers.size(), capture.n_embd};
    out.write(reinterpret_cast<char *>(header), sizeof(header));
    for (int tpos = 0; tpos < n_tokens; tpos++) {
        for (size_t li = 0; li < hf_layers.size(); li++) {
            auto it = std::find(capture.il_order.begin(), capture.il_order.end(), hf_layers[li] - 1);
            const std::vector<float> &b2 = capture.per_layer_buffers[it - capture.il_order.begin()];
            out.write(reinterpret_cast<const char *>(&b2[tpos * capture.n_embd]), capture.n_embd * sizeof(float));
        }
    }
    out.close();
    fprintf(stderr, "wrote %s: n_tokens=%d n_layers=%zu n_embd=%d\n", out_path.c_str(), n_tokens, hf_layers.size(), capture.n_embd);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

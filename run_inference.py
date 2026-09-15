import argparse
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast

# Model architecture
TOKEN_NUMBER = 32000
LAYER_NUMBER = 8
LONGUEUR_BLOC = 2048



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.num_heads = 8
        max_seq_len = LONGUEUR_BLOC
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        triangle_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("masque", triangle_mask.view(1, 1, max_seq_len, max_seq_len))
        self.linear_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        Batch, Temps, d_model = x.size()
        q = q.view(Batch, Temps, 8, d_model // 8).transpose(1, 2)
        k = k.view(Batch, Temps, 8, d_model // 8).transpose(1, 2)
        v = v.view(Batch, Temps, 8, d_model // 8).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / ((d_model / self.num_heads) ** 0.5)
        scores = torch.clamp(scores, min=-1e4, max=1e4)
        temp_mask = self.masque[..., :Temps, :Temps]
        scores = scores.masked_fill_(temp_mask == 0, float("-inf"))
        scores = nn.Softmax(dim=-1)(scores)
        attention = scores @ v
        output = attention.transpose(1, 2).contiguous().view(Batch, Temps, d_model)
        output = self.linear_out(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = MultiHeadAttention(d_model)
        self.feed_forward = FeedForward(d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_out = self.attention(self.layernorm1(x))
        x = x + attention_out
        x = x + self.feed_forward(self.layernorm2(x))
        return x


class MainModel(nn.Module):
    def __init__(self, vocab_size=TOKEN_NUMBER, d_model=512, num_couches=LAYER_NUMBER):
        super().__init__()
        self.pos_embedding = nn.Embedding(LONGUEUR_BLOC, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.couches = nn.ModuleList([TransformerBlock(d_model) for _ in range(num_couches)])
        self.final_norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        Batch, Temps = idx.size()
        pos = torch.arange(0, Temps, dtype=torch.long, device=idx.device)
        x = self.embedding(idx) + self.pos_embedding(pos)
        for couche in self.couches:
            x = couche(x)
        x = self.final_norm(x)
        logits = self.linear(x)
        return logits


# Utilities + generation


def load_tokenizer(path):
    import os
    from tokenizers import Tokenizer as TokenizersTokenizer

    path = os.path.expanduser(path)
    # If path is relative and doesn't exist, try resolving relative to this script's folder
    if not os.path.isabs(path) and not os.path.exists(path):
        script_dir = os.path.dirname(__file__)
        alt = os.path.join(script_dir, path)
        if os.path.exists(alt):
            path = alt

    # If a local folder was passed, prefer loading tokenizer.json directly
    if os.path.isdir(path):
        tokenizer_json = os.path.join(path, "tokenizer.json")
        if os.path.exists(tokenizer_json):
            tk = TokenizersTokenizer.from_file(tokenizer_json)
            return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|pad|>")
        # fallback: try transformers loader but force local files only to avoid hub validation issues
        return PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)

    # If a single file path was provided and exists, load it via tokenizers
    if os.path.exists(path):
        tk = TokenizersTokenizer.from_file(path)
        return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|pad|>")

    # Last resort: let transformers attempt to resolve (may download from hub)
    return PreTrainedTokenizerFast.from_pretrained(path)


def load_model(weights_path, device):
    import os
    # Resolve relative path against script dir if needed
    if not os.path.isabs(weights_path) and not os.path.exists(weights_path):
        script_dir = os.path.dirname(__file__)
        alt = os.path.join(script_dir, weights_path)
        if os.path.exists(alt):
            weights_path = alt

    model = MainModel()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def generate(model, tokenizer, prompt, device, max_new_tokens=128, ban_eos=True, temperature=0.0, top_k=None):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)  # (B, T, V)
            if ban_eos and eos_id is not None:
                logits[:, -1, eos_id] = -float("inf")

            if temperature == 0.0:
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                scores = logits[:, -1, :] / max(1e-8, temperature)
                if top_k is not None and top_k > 0:
                    topk_vals, topk_idx = torch.topk(scores, top_k)
                    probs = torch.zeros_like(scores).scatter(1, topk_idx, torch.softmax(topk_vals, dim=-1))
                else:
                    probs = torch.softmax(scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == eos_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run inference with FreakyLLM saved weights")
    parser.add_argument("--weights", default="Finished models/FreakyLLM.pth", help="Path to .pth weights")
    parser.add_argument("--tokenizer", default="Finished models/mon_tokenizer_perso", help="Tokenizer folder")
    parser.add_argument("--prompt", default="Once unpon a time", help="Prompt text") #prompt
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--no_ban_eos", action="store_true", help="Do not ban EOS token")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--interactive", action="store_true", help="Run in interactive REPL mode", default=True)
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive menu and run single generation")
    args = parser.parse_args()

    # Respect explicit --no-interactive
    if args.no_interactive:
        args.interactive = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.weights, device)

    # If interactive flag set, run a simple numbered menu in terminal
    if args.interactive:
        prompt = args.prompt
        max_new_tokens = args.max_new_tokens
        ban_eos = not args.no_ban_eos
        temperature = args.temperature
        top_k = args.top_k if args.top_k > 0 else None

        def show_settings():
            print('\nCurrent settings:')
            print(f' Prompt: "{prompt}"')
            print(f' Max new tokens: {max_new_tokens}')
            print(f' Ban EOS: {ban_eos}')
            print(f' Temperature: {temperature}')
            print(f' Top-k: {top_k}')

        try:
            while True:
                print('\nSimple menu:')
                print(' 1) Enter prompt')
                print(' 2) Generate now')
                print(' 3) Set max_new_tokens')
                print(' 4) Toggle ban EOS')
                print(' 5) Set temperature')
                print(' 6) Set top_k')
                print(' 7) Show settings')
                print(' 0) Exit')
                choice = input('Choose> ').strip()
                if choice == '0' or choice.lower() in ('q', 'quit', 'exit'):
                    break
                elif choice == '1':
                    prompt = input('Enter prompt: ').rstrip('\n')
                elif choice == '2':
                    print('\nGenerating...')
                    out = generate(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens, ban_eos=ban_eos, temperature=temperature, top_k=top_k)
                    print('\n=== Generated ===')
                    print(out)
                elif choice == '3':
                    try:
                        v = int(input('max_new_tokens: ').strip())
                        if v > 0:
                            max_new_tokens = v
                    except Exception:
                        print('Invalid number')
                elif choice == '4':
                    ban_eos = not ban_eos
                    print('Ban EOS set to', ban_eos)
                elif choice == '5':
                    try:
                        temperature = float(input('temperature (0 for greedy): ').strip())
                    except Exception:
                        print('Invalid number')
                elif choice == '6':
                    try:
                        k = int(input('top_k (0 to disable): ').strip())
                        top_k = k if k > 0 else None
                    except Exception:
                        print('Invalid number')
                elif choice == '7':
                    show_settings()
                else:
                    print('Unknown choice')
        except (KeyboardInterrupt, EOFError):
            print('\nExiting interactive menu')
    else:
        text = generate(
            model,
            tokenizer,
            args.prompt,
            device,
            max_new_tokens=args.max_new_tokens,
            ban_eos=not args.no_ban_eos,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )

        print('\n Generated')
        print(text)


if __name__ == "__main__":
    main()

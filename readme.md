# Attention Transformer Model
*Based on the foundational paper "Attention Is All You Need" (Vaswani et al., 2017), Google*


### Project Overview
While the original architecture was intended for state-of-the-art translation, I adapted it in this project to create a predictive text writer. My main objectives were:
- To create a a next-word predictor (similar to smartphone keyboard suggestions)
- To create a Story Generator from a starting sentence, testing the model's actual language understanding and context retention capacities.



### Architecture
*(Reference paper: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762))*


#### Model Structure
While the original paper relies on a full Encoder-Decoder structure for translation tasks, I modified the design to a **Decoder-only** autoregressive architecture (similar to modern LLMs like GPT). The entire model was built from scratch using PyTorch, without relying on pre-built Transformer layers.

Model mapping from the paper : 
<p align="center">
<img width="453" height="633" alt="image" src="https://github.com/user-attachments/assets/52b40601-45e8-44f1-9236-104dea00f727" />
</p>


#### Attention Function Mapping
To bridge the gap between theory and practice, I mapped the theoretical Multi-Head Attention mechanism directly to my custom PyTorch implementation:

<p align="center">
  <img width="45%" src="https://github.com/user-attachments/assets/fc65b82f-c21d-42ae-ae84-f103b2276e27" alt="Theoretical Attention from the paper" />
  &nbsp; &nbsp;
  <img width="45%" src="https://github.com/user-attachments/assets/235c47f7-ec0b-470c-ba30-948ac27185fb" alt="My PyTorch forward function" />
</p>
<p align="center">
  <em>Left: Original mechanism from the paper | Right: My PyTorch implementation</em>
</p>



### Training
For the training phase, I went with the following setup :

- **Dataset:** ~8GB of text data extracted from eBooks (primarily romance, entertainment, religion, and gaming genres).
- **Tokenization:** BPE (Byte-Pair Encoding) tokenizer, custom-configured with `<|pad|>` and eos (`<|endoftext|>`) special tokens.
- **Hyperparameters:** AdamW optimizer (`lr=5e-6`, `betas=(0.9, 0.98)`, `weight_decay=0.01`).

I trained the model for 12 hours  (over 400,000 steps) on a Nvidia NVIDIA RTX PRO 6000 Blackwell Workstation Edition, reaching a final loss of **1.39**. 

As shown in the loss evolution graph below, the curve shows no signs of stagnation yet, suggesting that a longer training time would likely yield even better performance.
<p align="center">
  <img width="80%" src="https://github.com/user-attachments/assets/ca3440c5-28ed-4dcf-a757-dd8cda450af2" alt="Loss evolution graph" />
</p>



### Results & Observations

Testing the model's capabilities revealed interesting insights into its language understanding and generation behavior:

The model successfully learned the statistical structure of English. It generates well-formed sentences and can maintain local context (coherence over 2 to 3 sentences). However, it lacks the long-term memory or understanding required to build an actual narrative storyline.

Using Greedy Inference (always picking the highest probability word), the model can quickly fall into repetitive text loops (which is not surprising for a raw autoregressive model). 
Increasing the heat breaks the loops and adds creativity, but pushes the model to hallucinate or generate nonsensical text with very little word to word grammatical/language accuracy.
During early testing, the model tended to generate the `<|endoftext|>` token too frequently, abruptly stopping generation after a few sentences. To evaluate longer outputs, to solve this and observe longer story generation i manually changed the EOS token's probability to `-inf` to force the model to keep writing without interfering too much into its thought processes.


#### Short Generation results (extract):
> **I went to the** band marches and put him down on. "I don't go." He turned to say in a few weeks. "You're going to him, skimming a few minutes." She wants good money out of me later." She's an open in his head. He looked at him away. She felt so bad glass said, but she'd invited me." Allen when she looked over and Allen back on her her out of her the door. She looked at him a couple of her like that very pot. "She looked over Allenose in a few days, she's breakfast, and went back." Mom. Rick about her, and she'd found the money out, and Allen's seat.

we note some language errors such as "she's breakfast" and the lost of meaning after a few sentences.


#### Evolution of the model (4 hours vs 12 hours):
An Interesting example is the difference of result when the model was trained for 4 hours compared to 12,

Here is the first generation (Loss : 2.6)
> **i went by** the train was the train station. The train station was not in the train was to the train of the train was not to the train was had been to the train station was was in the train was the train was in the train was in the train was the train was had been the train was in the train station of the train was the train was the train was had had had been was the train was the train was in the train

The Second generation (Loss : 1.3)
> **i went by** seeing so well before, and counts towards them, and were the long journey to be env-w them. The principal lessons they had not able to improve it was a few and left, uncomfort'clock at the widow and fatherless and other the great lakes, and other Indians and the great miles away. The answer was the other, but had the great looking around in the missionary came and began to meet the land to them had been from where the first Indian said there, they had some had a faithful with clay, they had to hear that point where the missionary. They had to talk,

We not the radical differences of language, and understanding abilities

<small>Note: These are short extracts of the output; each generation was capped at a total of 800 tokens.</small>



### Technical Challenges Overcome
Coding a Transformer entirely from scratch instead of just importing `nn.Transformer` meant I had to deal with a lot of low-level bugs. Here is what I had to fix:

- **`NaN` Loss Bug:** During early training runs, the model's loss would suddenly collapse to `NaN` (Not a Number). I traced this back to a poorly implemented padding mask inside the Multi-Head Attention block. The flawed mask was causing a division by zero during the Softmax operation. Fixing the mask's topology stabilized the training immediately.
- **Tensor management:** Managing tensor shapes across 8 attention heads (`Batch, Heads, Time, D_model / Heads`) required a lot of tracking. Additionally, implementing the causal lower-triangular mask correctly was critical to ensure the model couldn't "cheat" by looking at future tokens during training.
- **Memory Optimization:** To train on an 8GB dataset with a context window of 2048 tokens without triggering CUDA Out-Of-Memory (OOM) errors, I implemented dynamic block chunking, mixed precision (`torch.amp.autocast`), and gradient accumulation.



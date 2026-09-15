# Attention Transformer model 
## Based on the paper "Attention is all you need" (Google) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

### First notes

The original architecture was intended for "frontier" translation model, here in this project i used it to create a predictive text writer with the objective to
 - create a a next-word predictor model (like the one on smartphones keyboards)
 - create a Story generator from a starting sentence (to test the actual language understandings capacities of the model)


### Architecture
From https://arxiv.org/pdf/1706.03762

####Model Architecture
Using encoder decoder structure from the paper :
<img width="453" height="633" alt="image" src="https://github.com/user-attachments/assets/52b40601-45e8-44f1-9236-104dea00f727" />


####Attention function mapping
Map from the paper
<img width="651" height="355" alt="image" src="https://github.com/user-attachments/assets/fc65b82f-c21d-42ae-ae84-f103b2276e27" />

python implementation (foward function) :
<img width="574" height="548" alt="image" src="https://github.com/user-attachments/assets/235c47f7-ec0b-470c-ba30-948ac27185fb" />



### Training
Using a ~8Gb Dataset focused on ebooks focusing mainly on romance, entertainment, religious and games epubs,
Using BPE encoding with pad and eos special tokens.
using adamW optimizer with the following parameters : lr=5e-6, betas=(0.9, 0.98), weight_decay=0.01

I trained it on a Nvidia NVIDIA RTX PRO 6000 Blackwell Workstation Edition for 12 hours, for approximately more that 400 000 steps, with an end loss at 1.39
Note : based on the graph, better result could have been find if the training lasted longer (since there is no sings yet of stagnation on the graph)

Loss evolution graph
<img width="739" height="448" alt="Capture d’écran du 2026-09-14 20-36-08" src="https://github.com/user-attachments/assets/ca3440c5-28ed-4dcf-a757-dd8cda450af2" />

#### results
 good a writting things, the language is good, falls in loops easely on greedy inference, with more heat stark writting nonsence, still unable to create a actual story with storyline but can maintain contexte for a few sentences
 had to greedly make the prediction of the eos (end of text) token as i would generate it too ofter and stop its generation after 2-3 sentences

#### Issues encountered
 Had an issue with an NaN loss due to a padding mask wich was badly implemented
 Ran into a few issues trying to implement the architecture and understandig it

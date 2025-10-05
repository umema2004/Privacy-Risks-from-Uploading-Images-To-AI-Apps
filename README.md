# Implementation & Methodology  
## 1.  Objective and Scope
   
This implementation aims to reproduce and validate the key principles of the RobustFace paper (Sadu et al., 2024) using the CelebA dataset. Instead of replicating the exact restoration architecture (Blind Deblurring + Super-Resolution), a simplified but conceptually equivalent bilateral restoration pipeline was implemented. The focus was on verifying whether restoration preprocessing meaningfully reduces the impact of adversarial perturbations on downstream recognition or feature separability. 

## 2. Dataset Preparation 

● Dataset: CelebA face dataset with over 200 000 annotated face images. 

● Splits: Train = 162 770, Validation = 19 867, Test = 19 962 (from list_eval_partition.csv). 

● Input: Grayscale 128 × 128 pixels, normalized to [0, 1]. 

● Task: Binary classification of the “Smiling” attribute, used as a proxy for face-recognition performance. 

## 3. Model Architecture 

A lightweight convolutional model, SmallFaceNet, was trained from scratch. It consists of stacked Conv–BatchNorm–ReLU blocks with adaptive pooling and two fully connected layers. 
This was chosen for efficiency and interpretability rather than raw accuracy, mirroring the “model agnostic defense” philosophy of RobustFace. 

Optimizer: Adam (learning rate = 1e-3)    Batch size: 64    Dropout: 0.3    Device: CUDA 

Performance Summary 
SmallFaceNet | Validation Accuracy: 93.07 % | Final Training Loss : 0.175
<img width="555" height="317" alt="image" src="https://github.com/user-attachments/assets/ee40893e-f92c-4c6b-b9f8-79a4353f8356" />

## 4. Adversarial Attack Generation 
Two adversarial threat models were simulated: 
1. BIM (Basic Iterative Method) — gradient-based pixel perturbations (P-FGSM equivalent). 
2. FLM (Fast Landmark Manipulation) — geometric landmark-based distortions. 
Both attacks were applied on the validation/test splits, producing around 20 000 adversarial samples each. 

## 5. Restoration Pipeline 
The restoration module acted as a lightweight approximation of the RobustFace BL + SR stage: 

● Method: Bilateral filter with (d = 5, σColor = 75, σSpace = 75) 

● Purpose: Reduce high-frequency noise and approximate manifold projection. 

● Output: Restored adversarial images saved to /kaggle/working/restored_test. 
<img width="775" height="328" alt="image" src="https://github.com/user-attachments/assets/67ea34fd-2fbe-4e25-8946-f23aa892679d" />

## 6. Feature Extraction and Classification 
For evaluation, the Weighted Local Magnitude Pattern (WLMP) descriptor was implemented to 
capture texture and gradient-magnitude variations. 
● WLMP histograms (256-D) were computed for each image. 

● Features from clean, adversarial, and restored images were used to train standard 

classifiers: SVM (linear & RBF), Random Forest, and k-NN. 
This procedure measures adversarial detectability rather than recognition accuracy — consistent 
with testing whether restoration reduces the adversarial footprint.
<img width="787" height="761" alt="image" src="https://github.com/user-attachments/assets/549445a3-fe60-4aa0-a96b-ad26e23068dc" />

# Results and Comparative Analysis 
## 1. Training Performance 
The surrogate model achieved strong baseline accuracy (≈ 93 %), confirming it could learn stable 
identity-related features before attacks. 
The training curve (Figure 5.1) shows rapid convergence within 3 epochs and minimal overfitting. 
## 2. Adversarial Generation and Restoration Outcomes 
● BIM attacks: Introduced visible pixel noise; all 19 962 test images successfully perturbed. 

● FLM attacks: Produced plausible yet semantically altered faces (landmark warps). 

o Restoration: Processed all images in ≈ 50 seconds (> 390 images/s), restoring image 
smoothness and realism.

## 3. WLMP Classification Results
<img width="808" height="526" alt="image" src="https://github.com/user-attachments/assets/58a52ce8-95d6-4183-be0e-0b46940cb4f5" />
Interpretation: 
Across classifiers, restoration reduced adversarial detectability by roughly 10–13 % on average, 
implying that bilateral smoothing removed discriminative perturbation patterns. 
This confirms that the restoration step successfully “pushes” adversarial samples closer to the clean 
image manifold. 
## 4. Comparison with the Base Paper
<img width="886" height="290" alt="image" src="https://github.com/user-attachments/assets/999eca78-f460-4e95-b4c6-eadea6180284" />
While the paper measures recognition accuracy restoration, this implementation quantifies 
adversarial detectability reduction. Both the techniques indicate that restoration neutralizes attack 
artifacts. 
The observed trends match directionally: restoration improves robustness (or hides adversarial 
traces) in all cases. 
## 5. Sources of Discrepancy 
**Simplified Restoration**: The full RobustFace BL + SR network (blind deblurring + super-resolution) was replaced by a simple bilateral filter. Reduced restoration precision; smaller accuracy gains. 

**Grayscale Input**: CelebA images were converted to L-channel; color cues were lost. Texture descriptors (WLMP) became less discriminative.

**Different Metric**: The paper measured recognition accuracy; this work measured WLMP-based detectability. Quantitative values not directly comparable. 

**No Adaptive Attack Testing**: Adaptive re-optimization through the defense not implemented. Possibly overestimates robustness margins. 

**Lightweight CNN**: SmallFaceNet is simpler than state-of-the-art recognition models. Limits baseline robustness and restoration effects. 
## 6. Key Findings 
1. Qualitative Consistency: Both the paper and this implementation show restoration reduces 
adversarial distortions. 
2. Quantitative Gap: Smaller accuracy shifts stem from the simplified, non-learned restoration. 
3. Efficiency: The bilateral filter achieved ~390 images/s processing speed which is ideal for 
edge or client-side privacy defenses. 
4. WLMP Insight: Post-restoration WLMP histograms became smoother, indicating reduced 
adversarial texture contrast.
## 7. Overall Interpretation 
The reproduction confirms the core principle of RobustFace: adversarial robustness can be 
enhanced through image restoration that reprojects faces toward the natural manifold. 
Even a minimal bilateral filtering step demonstrated measurable mitigation, validating the 
conceptual model proposed by Sadu et al. (2024).

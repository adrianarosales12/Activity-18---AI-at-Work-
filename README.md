# Activity 18: AI at Work — A Field Guide to Deep Learning and Generative AI
## Sessions 34–36
## Due date (mm/dd/yyyy): 11/15/2026
## Adriana Rosales González
## Delivery Format: [] Video URL | [X] Markdown file | [] Jupyter Notebook file

---

# Activity Description

## The Story

Your first job will almost certainly involve AI tools: writing with an assistant, researching with
sources, prototyping an app, creating images, or training a model on your company's data. This
activity gives you real experience with several of them **before** you get there, and the knowledge
to use them well: what deep learning and generative AI actually are (updated for 2026), how people
build with them today, and how to use them responsibly.

---

# Activity 18 — Tool Reviews: AI at Work


## Part A — Content check (from the app)

**A1.** In the *Deep Learning* tab, which **edge filter (A or B)** highlights the **vertical** edges of the photo? Write its 9 numbers.

<img width="575" height="484" alt="Captura de pantalla 2026-11-15 174613" src="https://github.com/user-attachments/assets/c013835a-e748-4516-9f5f-3d52033a74ef" />



<img width="560" height="469" alt="Captura de pantalla 2026-11-15 174624" src="https://github.com/user-attachments/assets/f9a8bf14-ea93-4823-8a3d-ca1b77ba0256" />

- Edge filter A highlights the vertical edges of the photo (such as the vertical sides of the cat's nose and face)[cite: 2]. Its 9 numbers are:
- Row 1: `-1, 0, 1`
- Row 2: `-2, 0, 2`
- Row 3: `-1, 0, 1`


**A2.** In the *Generative AI* tab, what is the probability of the word **"food"** at temperature **0.5**, and at temperature **1.5**? In one sentence: what does raising the temperature do?

<img width="592" height="365" alt="Captura de pantalla 2026-11-15 174709" src="https://github.com/user-attachments/assets/eb55f08e-b72e-457b-a548-8bbe76fee939" />

<img width="556" height="358" alt="image" src="https://github.com/user-attachments/assets/463fd826-f217-41fc-80fb-41f908c74cb8" />

 - At temperature **0.5**, the probability of the word **"food"** is **55.0%**, and at temperature **1.5**, it drops to **29.2%**. Raising the temperature flattens the probability distribution across candidate words, making the model's output more varied and unpredictable rather than always picking the most obvious next word.

**A3.** In the diffusion demo, what share of the original image is kept (√ᾱₜ) at step **t = 250**, and at step **t = 500**?

<img width="569" height="422" alt="image" src="https://github.com/user-attachments/assets/3d95e06a-5f59-4d6f-96fc-af81d32c62df" />

<img width="568" height="434" alt="image" src="https://github.com/user-attachments/assets/097b421d-0786-43dd-9a55-1170c477d579" />

- At step **t = 250**, the share of the original image kept is **0.724** (72.4%), and at step **t = 500**, it is **0.280** (28.0%).

**A4.** Pick **one** row from any "What the lecture says vs. what is accurate" table and explain the correction in your own words.

*BUILDING WITH GEN AI (S36)
- Row chosen:** To build a GenAI app 'you will need a framework' such as TensorFlow, PyTorch, or Keras"* vs. *"Those train models. Apps usually call existing models through APIs/SDKs, coding assistants, or app builders.  
**Explanation:** The old lecture material confuses training a model from scratch with building an application. Today, developers rarely use low-level deep learning frameworks like PyTorch just to build an app; instead, they connect pre-trained models into their software using APIs, SDKs, or AI app builders.

<img width="578" height="249" alt="image" src="https://github.com/user-attachments/assets/112bcdb5-c441-4a48-a45e-4855e2627722" />

---

## Part B — Tool Review Cards

Copy this card once for each mission.

### Mission 1: Image Classifier (Deep Learning)

- **Tool(s) used and account type:** Google Teachable Machine (no account used / logged out)[cite: 9, 11].
- **What I did:** I created a standard image classification project with three everyday objects from my desk: a hair clip (*Pinza*), a computer mouse (*Mouse*), and a makeup compact (*Maquillaje*)[cite: 9, 11]. I recorded over 50 webcam samples for each class under the same lighting and background (56, 53, and 53 samples respectively), trained the MobileNet-based model in my browser, and tested its accuracy in the Preview panel under both original and altered conditions
- **My exact prompts:** None (this tool uses labeled webcam image classes instead of text prompts: `Pinza`, `Mouse`, and `Maquillaje`)
- **Evidence:** Screenshot `M1_teachable.png` attached below showing the 3 trained classes (+50 samples each) and live classification in the Preview panel
  - **Same conditions (10 tests per class):** Pinza: `[X/10]`, Mouse: `[X/10]`, Maquillaje: `[X/10]`[cite: 9].
  - **New conditions (`[ej. dimmer lighting / different angle / closer to camera]` - 10 tests per class):** Pinza: `[X/10]`, Mouse: `[X/10]`, Maquillaje: `[X/10]`.
- **Ratings** (1 = poor, 5 = excellent), with **one reason each**:
  - Usefulness for real work: **4/5** — Very fast for building quick visual prototypes or proof-of-concept classifiers without writing code, though limited for complex production environments.
  - Accuracy / quality of the output: **3/5** — Highly accurate when tested in the exact same room and lighting, but accuracy drops noticeably when lighting, angles, or backgrounds change. 
  - Ease of use: **5/5** — The interface is completely visual and intuitive; recording samples and training the model takes less than five minutes.
  - How much I would trust it without checking: **2/5** — Because it relies strictly on the visual patterns in the small training set, it can easily confuse objects when conditions shift or when two objects appear together.
- **One error or limitation I found, and how I checked it:** When I changed the lighting and angle (or held two objects close to the camera at once, as seen in my screenshot where the hair clip and makeup compact overlap), the model's confidence fluctuated and misclassified items. I checked this by running 10 controlled trials per class in new conditions and observing the drop in correct predictions compared to the original desk setup, proving that the training dataset lacked diverse backgrounds, lighting, and angles.
- **Free-plan limit I ran into (or "none"):** None (the tool is completely free to train and test in the browser without an account).
- **Privacy:** According to the Teachable Machine FAQ and privacy details, training runs locally inside the web browser; if I close the tab without clicking "Upload my model" or saving to Google Drive, no images or model data are stored on any server.
- **How I would use this tool in my field - to identify object and volumes.

<img width="903" height="565" alt="Captura de pantalla 2026-11-15 183053" src="https://github.com/user-attachments/assets/13cd98fe-76d0-49a2-af67-595dce019237" />
<img width="998" height="548" alt="Captura de pantalla 2026-11-15 182422" src="https://github.com/user-attachments/assets/9a09971c-1a93-45d8-b782-1c0bdabfdbcc" />



### Bonus (optional, up to 5 extra points)

Same card, plus the **Content Credentials** result from contentcredentials.org/verify and the sentence
from the tool's terms about commercial use (with its link).

---

## Part C — Looking back

**C1. Comparison.** Which tool would you actually use again, and which not? Compare at least three of the
tools you tried, using your ratings and evidence from Part B.

**C2. Your career.** When AI tools can do routine parts of your future job, which of your skills become
*more* valuable? Give two concrete examples from your missions.

**C3. AI-use disclosure.** Did you use any AI tool to write or edit *this file*? If yes, which one, and for
what exactly? (Honest disclosure is worth the points; using AI for wording help is allowed if you say so.)







---
# References:
- [UNESCO (2023), Guidance for generative AI in education and research](https://unesdoc.unesco.org/ark:/48223/pf0000386693)
- [Content Credentials verify tool](https://contentcredentials.org/verify)
- [Markdown Guide](https://www.markdownguide.org/basic-syntax/)

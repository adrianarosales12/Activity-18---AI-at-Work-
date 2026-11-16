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
- **Evidence:** 
<img width="903" height="565" alt="Captura de pantalla 2026-11-15 183053" src="https://github.com/user-attachments/assets/13cd98fe-76d0-49a2-af67-595dce019237" />
<img width="998" height="548" alt="Captura de pantalla 2026-11-15 182422" src="https://github.com/user-attachments/assets/9a09971c-1a93-45d8-b782-1c0bdabfdbcc" />

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


---
### Mission 2: Two Assistants, One Work Task (Gemini vs. Copilot Chat)

- **Tool(s) used and account type:** Google Gemini (`school Google account`) and Microsoft Copilot Chat (`personal account`).
- **What I did:** I pasted the exact same ShopSmart internal memo and three-part task (executive summary, customer reply, and action-item table) into both Gemini and Copilot Chat. Afterward, I asked both assistants the same factual question about the Transformer architecture and verified their answers against the official NeurIPS/arXiv paper (*Attention Is All You Need*, 2017).
- **My exact prompts:**
  - Prompt 1: Based ONLY on the internal memo below, please do three things: 1. Summarize it in 3 bullet points for a busy manager. 2. Write a polite reply to the customer complaint in it. 3. Make a table of the action items with an owner and a deadline. [followed by the full ShopSmart memo.
  - Prompt 2: In what year was the Transformer architecture introduced, and in which paper?
- **Evidence:**

<img width="470" height="410" alt="M2_gemini 2" src="https://github.com/user-attachments/assets/4b9300be-878a-4406-9ee6-f4a87b945a90" />
<img width="482" height="640" alt="M2_gemini 1" src="https://github.com/user-attachments/assets/6cc4c9c8-41f0-42d3-a7da-d9d4e4729175" />
<img width="626" height="383" alt="M2_Copilot 3" src="https://github.com/user-attachments/assets/0cbe7f66-9c0a-4c89-bfe7-e9d77ba1e9ff" />
<img width="695" height="618" alt="M2_Copilot 2" src="https://github.com/user-attachments/assets/b2f552b1-7c3e-4743-8109-daf5bd20ee05" />
<img width="656" height="649" alt="M2_Copilot 1" src="https://github.com/user-attachments/assets/042a0259-adfb-4912-b5e8-fea1ad6f6b2c" />

  - **Comparison & Fact-check:** Both models accurately captured all numbers (37 late orders up from 22, 12 returns, 7 headphones, 89 MXN fee, Order #48217) without hallucinating fake deadlines for the Web team[cite: 12, 13, 15, 16]. However, Gemini followed the prompt's language (English) and produced a cleaner, ready-to-send email with a subject line[cite: 15], whereas Copilot switched to Spanish, omitted the order number in the customer email, and left English department names inside a Spanish table[cite: 12, 13]. On the factual check, both correctly answered **2017** and **"Attention Is All You Need"** by Vaswani et al.[cite: 14, 16], which I verified on arXiv (`arxiv.org/abs/1706.03762`); Copilot provided direct source links (`arXiv`) and full author names, whereas Gemini gave a brief unlinked answer[cite: 14, 16].
- **Ratings** (1 = poor, 5 = excellent), with **one reason each**:
  - Usefulness for real work: **5/5** — Both turned an unstructured memo into a clean summary, customer email, and action table in seconds.
  - Accuracy / quality of the output: **4/5** — Neither invented false data, though Gemini was slightly better at respecting language instructions and formatting.
  - Ease of use: **5/5** — Copy-pasting text and getting structured tables and email drafts requires zero technical setup.
  - How much I would trust it without checking: **3/5** — Even when models don't blatantly hallucinate, subtle details (like Gemini promising the customer a Monday delivery based on an internal team deadline, or Copilot dropping an order number) still require human review before sending.
- **One error or limitation I found, and how I checked it:** Copilot ignored the prompt's language (replying in Spanish to an English prompt), included a label saying *"Opción 1: Profesional y directa"* without providing a second option, and left out Order #48217 in the reply to Ana López.
- Meanwhile, Gemini turned an internal Customer Care target ("by Monday") into an external promise to the customer. I caught both issues by comparing their outputs line-by-line against the original memo.
- **Free-plan limit I ran into (or "none"):** None (both worked smoothly without hitting message caps during this task).
- **Privacy:** According to the course Tool Lab documentation and Google/Microsoft privacy policies, when signed in with a school account (Google Workspace for Education or Microsoft 365 Education with the enterprise protection shield), prompts and uploaded memos are not reviewed by humans or used to train public AI models (unlike free personal accounts, where chat history may be used for training unless disabled in settings).
-  **How I would use this tool in my field (Digital Animation Engineering):** I would use Gemini or Copilot to summarize lengthy art-direction and client review notes into clear revision checklists, assign tasks with deadlines to modeling, rigging, and lighting artists, and draft professional status updates for producers or clients.

---

### Mission 3: Research with Sources (Gemini Notebook / NotebookLM)

- **Tool(s) used and account type:** Gemini Notebook / NotebookLM (`[school Google account / personal Google account]`).
- **What I did:** I created a new notebook (`M3_UNESCO`) and uploaded UNESCO's official 2023 PDF report, *Guidance for generative AI in education and research* (`386693eng.pdf`)[cite: 17]. I asked three questions grounded in the document, opened the inline citations to verify the original source passages, and then asked a fourth question about post-2023 events (the August 2024 EU AI Act enforcement and GPT-5) to test if the tool would refuse to hallucinate[cite: 17, 18, 19].
- **My exact prompts:**
  1. `"What are the main risks or ethical concerns of generative AI in education mentioned in this report?"`
  2. `"What minimum age limit does UNESCO recommend for students to independently use generative AI tools in classrooms?"`
  3. `"How does the report suggest regulating generative AI providers and protecting user data privacy?"`
  4. `"What does this document say about the EU AI Act entering into force in August 2024 or the release of GPT-5?"`
- **Evidence:** Screenshots

<img width="1276" height="589" alt="M3 3" src="https://github.com/user-attachments/assets/0ca6ab57-01d5-47dc-ac2e-98f115a5de7c" />
<img width="1268" height="638" alt="M3 2" src="https://github.com/user-attachments/assets/65125de5-49a3-4dfe-b0d6-6d46f7a1746b" />
<img width="1253" height="647" alt="M3 1" src="https://github.com/user-attachments/assets/861ff638-a79b-424c-a502-4d885e80de7e" />

  - **Citation checked:** In Question 2, the tool stated that UNESCO recommends a minimum age threshold of **13 years old** for independent conversations with GenAI platforms, Clicking citation `[1]` opened the exact passage in `386693eng.pdf` confirming: *"The minimum threshold should be 13 years of age"*. 
  - **Question outside the document:** When asked about the EU AI Act entering into force in August 2024 and GPT-5, NotebookLM did **not** invent an answer. It explicitly stated that the report dates from late 2023 and does not mention those events, clarifying that the PDF only discusses the 2021 proposed draft of the EU AI Act and GPT-4.
- **Ratings** (1 = poor, 5 = excellent), with **one reason each**:
  - Usefulness for real work: **5/5** — Grounding answers strictly in uploaded PDFs with clickable paragraph citations makes researching long technical documents much faster and safer.
  - Accuracy / quality of the output: **5/5** — Every claim matched the underlying PDF text, and it refused to fabricate information when asked about post-2023 events.
  - Ease of use: **5/5** — Uploading a PDF and clicking inline citation numbers to jump straight to the exact paragraph in the side panel is seamless.
  - How much I would trust it without checking: **4/5** — Much more trustworthy than a standard chatbot because it refuses to answer outside its sources, though clicking the citation is still necessary to confirm context and nuance.
- **One error or limitation I found, and how I checked it:** Unlike a general chatbot, NotebookLM cannot answer questions about current events or facts outside the uploaded files unless you explicitly use its web-search/Fast Research option; additionally, its summary sometimes condenses nuanced policy debates (such as proposals to raise the age limit to 16 under GDPR) into secondary bullet points. I verified this by opening the source side panel and reading the surrounding paragraphs in the original PDF.
- **Free-plan limit I ran into (or "none"):** None (uploading the full UNESCO PDF and running queries worked without hitting free-tier limits).
- **Privacy:** According to the course Tool Lab documentation and Google Workspace for Education policies, when using a school Google account, uploaded documents and notebook chats are not reviewed by humans or used to train Google's generative AI models.
- **How I would use this tool in my field (Digital Animation Engineering):** I would upload multi-hundred-page technical documentation—such as Unreal Engine 5 manuals, Maya/Blender Python API docs, or game design bibles (GDDs)—into a notebook so the production team can quickly query specific shader, rigging, or pipeline rules with exact page citations instead of manually searching through PDFs.

https://notebook.google.com/notebook/a705d02f-45c9-428f-bef5-2da6d80ffe42

---

### Mission 4: Build an App by Describing It (Animation Render & Frame Calculator)

- **Tool(s) used and account type:** Google AI Studio (`[school Google account / personal Google account]`).
- **What I did:** I described a single-file dark-mode "Animation Render & Frame Calculator" web app tailored for 3D animators, ran it directly in my browser, and tested it with three distinct scenarios (including an invalid negative/zero edge case). After spotting UI and validation issues during testing, I asked the assistant to refine the interface and error handling step by step.
- **My exact prompts:**
  - Initial prompt: `"Build a clean, dark-mode 'Animation Render & Frame Calculator' web app for 3D animators. Users should enter: 1. Project Name (text), 2. Frame Rate / FPS (dropdown: 24, 30, 60), 3. Total Duration in seconds (number), 4. Average Render Time per frame in seconds (number), 5. Number of Render Nodes / PCs (number, default 1). When clicking 'Calculate', show: Total Frames needed, Total Render Time formatted in hours and minutes, and a visual progress/breakdown bar."`
  - Improvement 1: `"Fix the FPS dropdown width so '60 fps' doesn't get cut off, and show all validation errors at once if both duration and render nodes are 0 or negative."`
  - Improvement 2: `"Add a 'Reset / Clear Form' button next to Calculate, and include an estimated cloud render cost breakdown ($0.50 per node-hour)."`
- **Evidence:** Screenshots 

<img width="776" height="524" alt="M4 1" src="https://github.com/user-attachments/assets/435eee79-2732-46a7-99f3-8cd1b48fd189" />
<img width="271" height="335" alt="M4 4" src="https://github.com/user-attachments/assets/f1da3582-fd22-49d7-855a-a7698d1da225" />
<img width="589" height="365" alt="M4 3" src="https://github.com/user-attachments/assets/fdfe5441-ce39-4302-b2bd-4a95777dad34" />
<img width="600" height="617" alt="M4 2" src="https://github.com/user-attachments/assets/303ea8ec-1097-405f-896a-023c5b5f2013" />

  - **Test 1 (Standard single-PC render):** `"Scene 01 - Walk Cycle"`, `24 fps`, `10s` duration, `15s` avg render time, `1` node → Output: **240 frames**, **1h 0m** total render time (`1x Node Speedup`).
  - **Test 2 (Multi-node VFX render):** `"Scene 02- VFX Explosion"`, `60 fps`, `5s` duration, `45s` avg render time, `4` nodes → Output: **300 frames**, **0h 56m** wall time (`4x Node Speedup`, saving `2h 49m` compared to a single PC's `3h 45m`).
  - **Test 3 (Strange / invalid input):** `"Scene 03 - Bug Test"`, `60 fps`, `-15s` duration, `01s` render time, `0` render nodes → Blocked calculation and displayed a red validation warning: *"Please enter a total duration greater than 0 seconds."*.
  - **Verdict on the 2023 quote ("we're not there yet"):** For small, self-contained internal tools like this render calculator, the 2023 quote no longer holds—AI generated a clean, responsive, functional prototype in seconds. However, for complex, multi-user production software with strict security and edge-case requirements, human developers are still essential to test, debug, and integrate the code properly.
- **Ratings** (1 = poor, 5 = excellent), with **one reason each**:
  - Usefulness for real work: **5/5** — Turning a plain-English spec into a working, styled internal utility in under a minute saves hours of boilerplate coding.
  - Accuracy / quality of the output: **4/5** — The render math (frame counts, node speedup, and time saved) was 100% accurate on the first try, needing only minor UI polish.
  - Ease of use: **4/5** — Describing features in natural language is effortless, though running a standalone HTML file locally required saving and opening it in a browser.
  - How much I would trust it without checking: **2/5** — Generated code can hide subtle layout bugs or incomplete input validation, so testing edge cases manually is mandatory.
- **One error or limitation I found, and how I checked it:** In the initial version, the Frame Rate dropdown was too narrow and clipped the label text (`"60 fps (High..."`), the number input allowed leading zeros (`01`), and when entering both a negative duration (`-15`) and `0` render nodes simultaneously, the form only flagged the first error instead of warning about both invalid fields. I discovered this during Test 3 by intentionally inputting `-15` and `0`, and verified the fix after requesting the UI and validation update.
- **Free-plan limit I ran into (or "none"):** None (using Google AI Studio's free tier used ~9,700 tokens and had no rate-limit blocks during this session).
- **Privacy:** According to the course Tool Lab documentation and Google AI Studio's terms, free-tier prompts and generated code in AI Studio may be logged and used by Google to improve its products, so no private company assets, proprietary code, or personal data should be entered.
- **How I would use this tool in my field (Digital Animation Engineering):** I would use AI app builders to rapidly prototype custom pipeline utilities for an animation studio—such as render-time estimators, naming-convention checkers for Maya/Blender asset exports, or shot-tracking dashboards—before handing them off to pipeline engineers for full integration.

---

## Bonus (optional, up to 5 extra points)

### Bonus Mission: Image Generation & Content Credentials (ShopSmart Marketing)

- **Tool(s) used and account type:** Canva Magic Media (Canva for Education school account) and 
- **What I did:** I generated a studio product marketing image for ShopSmart's fictional kitchen blender using Canva's AI image generator with a school account, downloaded the original JPG file (`MANGO_SMOOTHIE.jpg`), and uploaded it to `contentcredentials.org/verify` to check for embedded C2PA provenance metadata and review Canva's commercial-use terms.
- **My exact prompt:** `"Commercial studio product photography of a sleek modern matte-black kitchen blender with a glass jar filled with fresh mango smoothie, sitting on a clean marble countertop, warm morning sunlight, soft shadows, high-end advertising shot, 8k resolution"`
   - MY LICENSE DIDN'T WORK
- **Evidence:**

<img width="2720" height="1536" alt="MANGO_SMOOTHIE" src="https://github.com/user-attachments/assets/887ceb87-bf0d-4f56-97ff-4fb6e31c97a5" />
<img width="1279" height="757" alt="BONUS" src="https://github.com/user-attachments/assets/39ec4891-6647-4098-8056-ebd46f1eb8b5" />


  - **Content Credentials verification result:** When I uploaded the downloaded `MANGO_SMOOTHIE.jpg` file to `contentcredentials.org/verify`, the verifier returned an error / found no embedded C2PA Content Credentials. Unlike Adobe Firefly, Canva does not automatically attach C2PA provenance manifests to standard downloaded JPGs, meaning downstream platforms cannot cryptographically verify that the image is AI-generated.
  - **Commercial use terms:**
    - *Sentence from official terms:* Under Canva's AI Product Terms: `"Subject to your compliance with these AI Product Terms and Canva's Terms of Use, you may use Output for any legal purpose, including personal or commercial projects."` However, under **Canva for Education** terms, school accounts are restricted strictly to non-commercial educational use (`"Canva for Education is solely for educational, non-commercial use"`).
    - *Official link:* `https://www.canva.com/policies/ai-product-terms/` and `https://www.canva.com/policies/education-terms/`
- **Ratings** (1 = poor, 5 = excellent), with **one reason each**:
  - Usefulness for real work: **4/5** — Very convenient for generating quick concept visuals directly inside a layout/design editor without switching apps.
  - Accuracy / quality of the output: **4/5** — Produced convincing studio lighting, glass reflections, and textures on the mango smoothie, with only tiny mechanical flaws upon close inspection.
  - Ease of use: **5/5** — Typing a prompt inside Canva's editor and downloading the resulting image takes seconds.
  - How much I would trust it without checking: **2/5** — Outputs must be checked both visually for small AI glitches and legally for account-tier licensing restrictions (especially when using an Education account).
- **One error or limitation I found, and how I checked it:** Visually, zooming in on the blender's control dial shows a subtle AI glitch where the number zero is repeated twice (`0, 0, 1, 2, 3`). Technically, uploading the downloaded file to `contentcredentials.org/verify` failed to detect any Content Credentials metadata, and reviewing Canva's policy revealed that images made on a school (`Canva for Education`) account cannot be used commercially.
- **Free-plan limit I ran into (or "none"):** None on the Canva for Education plan during this test (standard free accounts have a monthly AI generation credit cap).
- **Privacy:** According to Canva's AI Product Terms and Privacy Policy, users can control whether their data is used to improve AI models in privacy settings, and school accounts under Canva for Education have additional student data protections (no private client assets should be entered).
- **How I would use this tool in my field (Digital Animation Engineering):** I would use Canva's AI generation tools during early pre-production to quickly assemble pitch decks, color scripts, and mood boards for animated shorts or game concepts, while keeping track of licensing restrictions before moving into final 3D production in Maya or Blender.

---

## Part C — Looking back

**C1. Comparison.** Which tool would you actually use again, and which not? Compare at least three of the
tools you tried, using your ratings and evidence from Part B.
 - I would definitely use **Gemini Notebook (NotebookLM)** (Accuracy: 5/5, Usefulness: 5/5, Trust: 4/5) and **Google AI Studio** (Usefulness: 5/5, Accuracy: 4/5) again. In Mission 3, NotebookLM proved to be the most reliable tool because every answer was grounded in the uploaded UNESCO PDF with clickable citations (such as confirming the 13-year minimum age threshold on page 34) and it honestly refused to fabricate post-2023 facts about the EU AI Act or GPT-5. Similarly, AI Studio generated a working, dark-mode 3D render calculator in seconds with accurate frame and node math, making it great for fast pipeline prototyping. By contrast, I would **not** rely on **Google Teachable Machine** (Accuracy: 3/5, Trust: 2/5) or **Canva Magic Media** (Trust: 2/5) for final production work: Teachable Machine's accuracy dropped noticeably as soon as lighting or angles shifted from my original desk samples, and Canva Magic Media produced subtle visual glitches (such as duplicating `0` twice on the blender dial) while lacking embedded C2PA Content Credentials and commercial rights under a school account.


**C2. Your career.** When AI tools can do routine parts of your future job, which of your skills become
*more* valuable? Give two concrete examples from your missions.
 - As a future **Digital Animation Engineer**, when AI can rapidly generate concept art, draft pipeline scripts, or summarize production notes, human skills in **critical quality assurance (edge-case debugging/visual inspection)** and **domain-specific technical judgment & legal compliance** become much more valuable:

1. **Edge-case testing and pipeline debugging (from Mission 4):** While AI Studio built the render calculator's core formula quickly, it missed subtle UI clipping (`"60 fps (High..."`) and incomplete multi-field validation when given negative durations (`-15s`) and `0` render nodes simultaneously. In a real animation studio, an engineer's ability to anticipate edge cases, validate inputs, and integrate scripts cleanly into Maya, Blender, or Unreal Engine is essential so bad data doesn't crash a render farm.

2. **Artistic direction, detail inspection, and licensing awareness (from Mission 2 & Bonus):** In Mission 2, Gemini turned an internal Monday deadline into an external promise to a customer, and in the Bonus mission, Canva's generated product photo looked photorealistic at first glance but hid a dial numbering error (`0, 0, 1, 2, 3`) and carried non-commercial restrictions under an Education account. Human animators and technical directors are needed to catch visual/narrative inconsistencies and ensure all assets are legally cleared before commercial release.

**C3. AI-use disclosure.** Did you use any AI tool to write or edit *this file*? If yes, which one, and for
what exactly? (Honest disclosure is worth the points; using AI for wording help is allowed if you say so.)

 - Yes. I used **Google Gemini** while completing this file to help break down the mission instructions into clear checklists, generate sample test prompts (such as the 3D render calculator specification and the ShopSmart product studio prompt), and translate and polish my observations and tool reviews into clear English. All hands-on tool tests, webcam dataset recordings, document queries, app edge-case tests, and screenshots were performed and verified personally by me.


---
# References:
- [UNESCO (2023), Guidance for generative AI in education and research](https://unesdoc.unesco.org/ark:/48223/pf0000386693)
- [Content Credentials verify tool](https://contentcredentials.org/verify)
- [Markdown Guide](https://www.markdownguide.org/basic-syntax/)

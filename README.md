<p align="center">
  <img width="280px" src="./assets/facial-recognition.png">
</p>
<p align="center" dir="auto">
  <img align="middle" width="32px" src="./assets/face-scan.png"/> 
  DMAD - Differential Morphing Attack Detection.
 <img align="middle" width="32px" src="./assets/face-scan.png"/> <br><br>
  A project for Fundamentals of Computer Vision and Biometrics <br>
  course at the University of Salerno.
</p>

<h1 dir="auto">
  <a id="user-content-project-description-and-introduction" class="anchor" aria-hidden="true" href="#project-description-and-introduction">
    <svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z">
      </path>
    </svg>
  </a>
  Project description and introduction
</h1>
<p dir="auto">In this section we introduce context informations for the project.</p>
<h2 dir="auto">
  <a id="user-content-introduction" class="anchor" aria-hidden="true" href="#introduction"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true">
    <path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z">
    </path>
    </svg>
  </a>
  Introduction
</h2>
<p dir="auto">
  🧠 Facial Recognition (FR) systems are vulnerable to multiple attacks, one of which is the morphing attack of the face (MA). Therefore, we have decided to develop a classifier capable of recognizing several of these attacks based on five different techniques: OpenCV (OCV), FaceMorpher (FM), Style-GAN 2 (SG), WebMorpher (WM), and AMSL. The model used, MixNets, was trained on the SMDD dataset and validated on subsets of the test set.
</p>

<p dir="auto">
🧠 After replicating the experimentation proposed in the <a href="./paper/paper analizzato.pdf"> paper</a>, we conducted additional checks. Specifically, we compared MixNets with the geometric approach, which manipulates the geometry of the faces. This comparison aims to determine which of the two approaches performs better. Moreover, our goal is to investigate if combining the two approaches can yield improved metrics. We are also interested in examining whether MixNets can enhance the critical points of the geometric approach in recognizing smiling morphs, as it tends to struggle with texture changes and complex details such as smiles.
</p>
 
<p dir="auto">
🧠 To enable this comparison, the first step is to extract the features from the penultimate layer of the network. These features are then subjected to preprocessing using PCA. The resulting output is fed into the classifiers to obtain initial results. This process is repeated for both the test sets containing smiles and those without. Subsequently, we merge the .csv files containing the features of the two approaches. Two merges are performed: one with smiles and one without.
</p>

<h2 dir="auto">
  <a id="user-content-documentation" class="anchor" aria-hidden="true" href="#documentation">
    <svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true">
      <path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z">
      </path>
    </svg>
  </a>
  Documentation
</h2>
<ul dir="auto">
<li>Our paper can be found in <code>paper</code> directory at the following link: <a href="/paper/paper analizzato.pdf" rel="nofollow">Paper Differential-MAD</a></li><br>
<li>The paper analyzed can be found in <code>paper</code> directory at the following link: <a href="/paper/paper analizzato.pdf" rel="nofollow">
  <br>Privacy-friendly Synthetic Data for the Development of Face MAD</a></li><br>
<li>MixNet paper can be found in <code>paper</code> directory at the following link: <a href="./paper/MixNet-paper.pdf" rel="nofollow">MixNet</a>.</li><br>
<li>Statistics graphics can be found in <code>code/graphics</code> directory at the following link: <a href="./code/graphics" rel="nofollow">Statistics graphics</a>.</li>
</ul>
<h1 dir="auto">
  <a id="user-content-technical-informations" class="anchor" aria-hidden="true" href="#technical-informations">
    <svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true">
      <path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z">
      </path>
    </svg>
  </a>
  Technical informations
</h1>
<p dir="auto">Requirements can be found in <code>code</code> directory at the following link: <a href="./code/requirements.txt" rel="nofollow">requirements</a>
</p>

<h2 dir="auto">
  <a id="user-content-authors" class="anchor" aria-hidden="true" href="#authors">
    <svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z">
      </path>
    </svg>
  </a>
  Author & Contacts
</h2>


| Name | Description |
| --- | --- |
| <p dir="auto"><strong>Alberto Montefusco</strong> |<br><em>Developer</em> - <a href="https://github.com/Alberto-00">Alberto-00</a></p><p dir="auto">Email - <a href="mailto:a.montefusco28@studenti.unisa.it">a.montefusco28@studenti.unisa.it</a></p><p dir="auto">LinkedIn - <a href="https://www.linkedin.com/in/alberto-montefusco">Alberto Montefusco</a></p><p dir="auto">My WebSite - <a href="https://alberto-00.github.io/">alberto-00.github.io</a></p><br>|
| <p dir="auto"><strong>Alessandro Aquino</strong> |<br><em>Developer</em>   - <a href="https://github.com/AlessandroUnisa">AlessandroUnisa</a></p><p dir="auto">Email - <a href="mailto:a.aquino33@studenti.unisa.it">a.aquino33@studenti.unisa.it</a></p><p dir="auto">LinkedIn - <a href="https://www.linkedin.com/in/alessandro-aquino-62b74218a/">Alessandro Aquino</a></p><br>|
| <p dir="auto"><strong>Simone Tartaglia</strong> |<br><em>Developer</em>   - <a href="https://github.com/drybonez01">drybonez01</a></p><p dir="auto">Email - <a href="mailto:s.tartaglia3@studenti.unisa.it">s.tartaglia3@studenti.unisa.it</a></p><br>|

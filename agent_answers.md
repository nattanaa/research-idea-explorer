## Idea 1: Can large language models catch errors in clinical trial protocols?

Existing work covers using large language models to assess clinical trial protocols for regulatory compliance, but we did not find work on using large language models to catch errors in clinical trial protocols. The closest paper is very close, evaluating how a large language model reviews clinical trial protocols against regulatory guidance [Large Language Models for Clinical Trial Protocol Assessments](https://openalex.org/W4415416600). This preliminary assessment is based on a search of a few queries reading abstracts and summaries only, yielding 22 papers cited in the findings.

### Closest existing work

This paper evaluated the performance of a large language model in reviewing specific components of clinical trial protocols against regulatory guidance, which differs from your idea because it focuses on compliance standards rather than using large language models to catch errors [Large Language Models for Clinical Trial Protocol Assessments](https://openalex.org/W4415416600).

This study developed a workflow to translate narrative clinical trial protocols into formal logic formulas using large language models for automated reasoning, which differs from your idea because it focuses on formal logical translation rather than direct methods for large language models to catch errors [Scaling Up Formal Representation of Clinical Trial Protocols in Ensemble Logic Using LLMs: A Preliminary Study](https://openalex.org/W7170638918).

This research tested retrieval-augmented generation and large language model systems on clinical trial protocols to evaluate compliance with regulatory documents, which differs from your idea because it evaluates external guidance compliance instead of catching internal errors [Retrieval Augmented Generation (RAG) for Evaluating Regulatory Compliance of Drug Information and Clinical Trial Protocols](https://openalex.org/W7130503162).

This work introduced a benchmark and evaluation framework to measure the compliance and factuality of clinical research documents generated from protocols by large language models, which differs from your idea because it focuses on evaluating generated outputs rather than using large language models to catch errors in the source clinical trial protocols [Compliance and factuality of large language models for clinical research document generation](https://openalex.org/W4415605138).

### Already well covered

Existing research extensively covers using large language models to assess clinical trial protocols for regulatory compliance and statistical design reviews [Large Language Models for Clinical Trial Protocol Assessments](https://openalex.org/W4415416600). For instance, researchers have assessed retrieval-augmented generation systems on clinical trial protocols to check regulatory compliance [Retrieval Augmented Generation (RAG) for Evaluating Regulatory Compliance of Drug Information and Clinical Trial Protocols](https://openalex.org/W7130503162). Additionally, using large language models to translate narrative clinical trial protocols into structured logic representations for reasoning is well documented [Scaling Up Formal Representation of Clinical Trial Protocols in Ensemble Logic Using LLMs: A Preliminary Study](https://openalex.org/W7170638918).

### Gaps

- We did not find large language models catching errors in clinical trial protocols.

### Who to talk to

- **Murali Krishna Ramanathan** (University at Buffalo, State University of New York) was the last author of a study assessing clinical trial protocols against regulatory guidelines using large language models [Large Language Models for Clinical Trial Protocol Assessments](https://openalex.org/W4415416600). This covers the clinical trial protocol assessment side of your idea. You can ask them: "How can the rubrics you used to assess statistical and pharmacokinetic components be adapted to help large language models catch errors in clinical trial protocols?"
- **Guo‐Qiang Zhang** (The University of Texas Health Science Center at Houston) was the last author of research translating clinical trial protocols into ensemble logic formulas using large language models [Scaling Up Formal Representation of Clinical Trial Protocols in Ensemble Logic Using LLMs: A Preliminary Study](https://openalex.org/W7170638918). This covers the logical validation side of your idea. You can ask them: "Can translating clinical trial protocols into ensemble logic formulas be used to catch errors in clinical trial protocols?"
- **Chunhua Weng** (Columbia University) was the last author of a study establishing a knowledge base of clinical trial eligibility criteria derived from parsed clinical trial protocols [A knowledge base of clinical trial eligibility criteria](https://openalex.org/W3149151080). This covers the protocol parsing side of your idea. You can ask them: "How can structured eligibility criteria databases help large language models catch errors in clinical trial protocols?"
- **Jimeng Sun** (University of Illinois Urbana-Champaign) was the last author of a study introducing a compliance and factuality evaluation framework for clinical research documents generated from protocols [Compliance and factuality of large language models for clinical research document generation](https://openalex.org/W4415605138). This covers the evaluation and benchmarking side of your idea. You can ask them: "Can your compliance and factuality framework be modified to evaluate how well large language models catch errors in clinical trial protocols?"

### Limits of this search

We searched the arxiv, openalex, pubmed, and semantic_scholar databases. The semantic_scholar database failed in 2 of 9 searches due to an HTTP 429 Too Many Requests error, though it succeeded in the others. Only titles, abstracts, and summaries were read; we did not read any full papers. The searches for automated error-catching systems, rule-based NLP parsers for internal inconsistencies, and multi-agent protocol self-correction did not return any papers to open and rest on search summaries only. The candidates listed are the authors of the closest papers found rather than a comprehensive measure of the leading experts in the field, and their listed affiliations may be out of date.

## Evidence behind the expert ranking

| # | Researcher | Latest listed affiliation | Found via angle | Cited papers | First/last author on | Latest paper |
|---|---|---|---|---|---|---|
| 1 | [Murali Krishna Ramanathan](https://openalex.org/A5002609733) | University at Buffalo, State University of New York | Find existing work that uses large language models to asses… | 3 | 3 | 2026 |
| 2 | [Guo‐Qiang Zhang](https://openalex.org/A5100392487) | The University of Texas Health Science Center at Houston | Analyze how formal logic or multi-agent frameworks are used… | 1 | 1 | 2026 |
| 3 | [Chunhua Weng](https://openalex.org/A5009604048) | Columbia University | Investigate traditional or rule-based NLP techniques for ch… | 3 | 3 | 2021 |
| 4 | [Arturo Loaiza‐Bonilla](https://openalex.org/A5060268370) | St. Luke's University Health Network | Analyze how formal logic or multi-agent frameworks are used… | 1 | 1 | 2026 |
| 5 | [Alban Bornet](https://openalex.org/A5017831867) | University of Geneva | Determine how LLM performance in auditing or assessing clin… | 1 | 1 | 2025 |
| 6 | [Jimeng Sun](https://openalex.org/A5084279065) | University of Illinois Urbana-Champaign | Determine how LLM performance in auditing or assessing clin… | 1 | 1 | 2025 |
| 7 | [Halil Kilicoglu B](https://openalex.org/A5016571803) | University of Illinois Urbana-Champaign | Investigate traditional or rule-based NLP techniques for ch… | 1 | 1 | 2025 |
| 8 | [Honghao Lai](https://openalex.org/A5085639088) | Lanzhou University | Find existing work that uses large language models to asses… | 1 | 1 | 2024 |

Chosen and ranked by code, not by a model, from 22 of the 22 papers cited above (authors matched by OpenAlex id). Each angle of the search contributed its best candidates, and at most one person is listed per paper. This shows who wrote the closest papers we found, not who is the leading expert. Check each profile for current affiliation before contacting anyone.

---
_28 tool calls: {"Lead": 7, "Researcher 1": 5, "Researcher 2": 6, "Researcher 3": 5, "Researcher 4": 5}_

## Idea 2: Graph neural networks for predicting drug-drug interactions in patients taking five or more medications

Existing work extensively covers using graph neural networks to predict drug-drug interactions [Modeling polypharmacy side effects](https://openalex.org/W2786016794). However, we did not find work evaluating graph neural networks for predicting drug-drug interactions in patients taking five or more medications. The closest paper is a conceptual framework that proposes utilizing graph neural networks to predict drug-drug interactions in patients taking five or more medications [Graph Neural Networks for Drug-Drug Interaction Prediction](https://openalex.org/W7211911824). This preliminary search was limited, covering only a few queries and the abstracts of the 17 papers cited in the findings.

# Closest existing work

This work proposed a conceptual framework utilizing graph neural networks to integrate prescription sequences and molecular structures to predict drug-drug interactions in polypharmacy patients taking five or more medications [Graph Neural Networks for Drug-Drug Interaction Prediction](https://openalex.org/W7211911824). It is highly similar to your idea but differs by presenting a conceptual framework rather than a fully evaluated empirical implementation [Graph Neural Networks for Drug-Drug Interaction Prediction](https://openalex.org/W7211911824).

This study developed Decagon, a multimodal graph convolutional neural network to model polypharmacy side effects by predicting exact drug-drug interaction types in large multi-relational networks [Modeling polypharmacy side effects](https://openalex.org/W2786016794). It differs from your idea because it focuses on predicting pairwise drug-drug interactions rather than predicting interactions specifically for patients taking five or more medications [Modeling polypharmacy side effects](https://openalex.org/W2786016794).

This paper proposed a graph co-attentional neural network architecture that integrates molecular structures and side-effect types early to predict adverse drug-drug interactions for drug combinations [Drug-Drug Adverse Effect Prediction](https://openalex.org/W2943714799). It differs from your idea as it does not explicitly restrict or condition its predictions on patient cohorts taking five or more medications [Drug-Drug Adverse Effect Prediction](https://openalex.org/W2943714799).

This research developed a sparse hypergraph neural network (SPARSE) to learn multiple types of latent combinations and capture higher-order relationships among interacting drugs and side effects [SPARSE](https://openalex.org/W4225351203). It differs from your idea by using hypergraph neural networks to learn latent combinations of drugs rather than applying graph neural networks to predict drug-drug interactions for patients taking five or more medications [SPARSE](https://openalex.org/W4225351203).

# Already well covered

Existing research has extensively focused on using graph neural networks to predict pairwise drug-drug interactions and polypharmacy side effects [Modeling polypharmacy side effects](https://openalex.org/W2786016794) [Drug-Drug Adverse Effect Prediction](https://openalex.org/W2943714799). Researchers have also successfully integrated molecular structures, multi-modal features, and knowledge graphs with graph neural networks to enhance predictions of drug-drug interactions [KGNN](https://openalex.org/W3035011799) [GAINET](https://openalex.org/W4407425624) [Advancing Drug-Drug Interaction Prediction](https://openalex.org/W7126086195). Furthermore, several studies have utilized hypergraph structures to model higher-order drug combinations and complex relationships [SPARSE](https://openalex.org/W4225351203) [Interactive multi-hypergraph](https://openalex.org/W4404128140).

# Gaps

- We did not find evaluations of graph neural networks for predicting drug-drug interactions specifically in patients taking five or more medications.
- We did not find graph neural network models designed to predict drug-drug interactions for patients taking five or more medications, as existing graph models focus primarily on pairwise drug-drug interactions.

# Who to talk to

- **Daniel Fischer** (Karlsruhe Institute of Technology): They were the first author on the paper that proposed a conceptual framework utilizing graph neural networks to integrate prescription sequences and molecular structures to predict drug-drug interactions in polypharmacy patients taking five or more medications [Graph Neural Networks for Drug-Drug Interaction Prediction](https://openalex.org/W7211911824). This covers the side of your idea involving modeling drug-drug interaction predictions specifically in polypharmacy patients taking five or more medications [Graph Neural Networks for Drug-Drug Interaction Prediction](https://openalex.org/W7211911824). 
  *Question to ask:* How can your conceptual framework integrating prescription sequences and molecular structures be transitioned into a fully evaluated empirical model for patients taking five or more medications?

- **Jure Leskovec** (Chan Zuckerberg Biohub San Francisco): They were the last author on the paper that developed a multimodal graph convolutional neural network to model polypharmacy side effects by predicting exact drug-drug interaction types in large multi-relational networks [Modeling polypharmacy side effects](https://openalex.org/W2786016794). This covers the side of your idea involving predicting drug-drug interactions using graph neural networks [Modeling polypharmacy side effects](https://openalex.org/W2786016794). 
  *Question to ask:* How can multi-relational graph convolutional networks, like the one designed for Decagon, be adapted to predict drug-drug interactions for patients taking five or more medications?

- **Nansu Zong** (Mayo Clinic): They were the last author on the paper that combined graph neural networks and transformers to transform electronic health records into graph structures to model patient subgroups and optimize drug response predictions across multiple medication categories [Stratifying heart failure patients](https://openalex.org/W4400100984). This covers the side of your idea involving predicting drug responses and modeling patient-level data using graph neural networks [Stratifying heart failure patients](https://openalex.org/W4400100984). 
  *Question to ask:* What are the primary challenges in structuring patient records to predict drug-drug interactions specifically for patient subgroups taking five or more medications?

- **Oliver Lloyd** (University of Bristol): They were the first author on the paper that demonstrated that properly optimized tensor factorization models can efficiently predict polypharmacy side effects across large sets of drug combinations [Fast polypharmacy side effect prediction](https://openalex.org/W4404704751). This covers the side of your idea involving predicting interactions and adverse events in multi-drug combinations [Fast polypharmacy side effect prediction](https://openalex.org/W4404704751). 
  *Question to ask:* How do tensor factorization methods compare with graph neural networks in terms of computational efficiency when predicting drug-drug interactions for patients taking five or more medications?

# Limits of this search

The databases searched for this report were arXiv, OpenAlex, PubMed, and Semantic Scholar. No database failed during these searches. Only titles, abstracts, and summaries were read, and full papers were not analyzed. The listed candidates are the authors of the closest papers found rather than a comprehensive measure of the leading experts in the field, and their affiliations may be out of date.

## Evidence behind the expert ranking

| # | Researcher | Latest listed affiliation | Found via angle | Cited papers | First/last author on | Latest paper |
|---|---|---|---|---|---|---|
| 1 | [Jure Leskovec](https://openalex.org/A5091272738) | Chan Zuckerberg Biohub San Francisco | Identify existing work using graph neural networks to predi… | 1 | 1 | 2018 |
| 2 | [Bihter Daş](https://openalex.org/A5054379190) | Fırat University | Identify existing work using graph neural networks to predi… | 1 | 1 | 2025 |
| 3 | [Nansu Zong](https://openalex.org/A5019653651) | Mayo Clinic | Examine how patient-level clinical records of polypharmacy … | 1 | 1 | 2024 |
| 4 | [Amir Lakizadeh](https://openalex.org/A5033018744) | University of Qom | Identify existing work using graph neural networks to predi… | 1 | 1 | 2024 |
| 5 | [Tiangang Zhang](https://openalex.org/A5004963500) | Hainan University | Investigate graph-based or machine learning methods for pre… | 1 | 1 | 2024 |
| 6 | [Andreea Deac](https://openalex.org/A5013115798) | University of Cambridge | Identify existing work using graph neural networks to predi… | 1 | 1 | 2019 |
| 7 | [Oliver Lloyd](https://openalex.org/A5034502841) | University of Bristol | Investigate graph-based or machine learning methods for pre… | 1 | 1 | 2024 |
| 8 | [Daniel Fischer](https://openalex.org/A5048857636) | Karlsruhe Institute of Technology | Examine how patient-level clinical records of polypharmacy … | 1 | 1 | 2022 |

Chosen and ranked by code, not by a model, from 17 of the 17 papers cited above (authors matched by OpenAlex id). Each angle of the search contributed its best candidates, and at most one person is listed per paper. This shows who wrote the closest papers we found, not who is the leading expert. Check each profile for current affiliation before contacting anyone.

---
_24 tool calls: {"Lead": 6, "Researcher 2": 7, "Researcher 1": 6, "Researcher 3": 5}_

## Idea 3: Does remote work increase loneliness among employees at large companies?

Existing work covers how remote work affects workplace loneliness among employees, with the closest studies directly surveying remote workers to evaluate their loneliness. However, we did not find research specifically examining whether this relationship is affected by working at large companies. This assessment is based on a limited search of a few queries, reviewing only titles, abstracts, and summaries across the 18 papers cited in the findings.

### Closest existing work

This study evaluated workplace loneliness during remote work using a two-wave survey of remote employees, finding that job demands increase loneliness through work-home interference [Lonely@Work@Home?](https://openalex.org/W4372342495). It differs from your idea because it focuses on job and home demands as drivers of loneliness, rather than examining the role of large companies [Lonely@Work@Home?](https://openalex.org/W4372342495).

This paper explored the relationship between remote e-working and flow experience via a survey of 202 employees, demonstrating that remote work introduces technostress and loneliness as serial mediators [An examination of remote e-working](https://openalex.org/W3199758503). It differs from your idea because it looks at technostress and flow experience among general remote employees rather than focusing on employees at large companies [An examination of remote e-working](https://openalex.org/W3199758503).

This research investigated how working from home interacts with workplace loneliness to affect organizational commitment and coworker support using self-determination theory [Workplace Loneliness](https://openalex.org/W4281655115). It differs from your idea because it focuses on the interactive effects of working from home and loneliness on commitment and support, but does not investigate how working at large companies impacts employee loneliness [Workplace Loneliness](https://openalex.org/W4281655115).

### Already well covered

Research has examined how remote work introduces loneliness and how job demands and support influence workplace loneliness [Lonely@Work@Home?](https://openalex.org/W4372342495). Existing work has also investigated how remote work and workplace loneliness interact to affect employee commitment and coworker support [Workplace Loneliness](https://openalex.org/W4281655115). 

### Gaps

* We did not find empirical studies evaluating remote work and loneliness among employees specifically at large companies.
* We did not find research analyzing whether working at large companies moderates the effect of remote work on employee loneliness.

### Who to talk to

* **Timo Walz** (Maastricht University): They were the first author on [Lonely@Work@Home?](https://openalex.org/W4372342495), which was found through the search angle identifying the empirical relationship between remote work and employee loneliness. They cover the remote work and loneliness aspect of your idea. 
  * *Question to ask:* "In your survey of remote employees, did you collect information on company size, and do you think employees at large companies experience different levels of workplace loneliness compared to those at smaller organizations?"
* **Amy Wax** (California State University, Long Beach): They were the first author on [Workplace Loneliness](https://openalex.org/W4281655115), which was found through the search angle investigating workplace loneliness and employee connections in organizations. They cover the employee connection and remote work side of your idea.
  * *Question to ask:* "Based on your findings regarding working from home and coworker support, how might the structural complexity of a large company influence how remote work affects employee loneliness?"
* **Didem Taser Erdogan** (Brunel University of London): They were the first author on [An examination of remote e-working](https://openalex.org/W3199758503), which was found through the search angle identifying the empirical relationship between remote work and workplace loneliness. They cover the remote e-working and loneliness side of your idea.
  * *Question to ask:* "How might the technostress and loneliness experienced by remote employees differ when operating within a large company compared to smaller firms?"
* **Kristen Faile** (Belmont University): They were the first author on [Reducing Workplace Loneliness](https://openalex.org/W4409488219), which was found through the search angle evaluating how remote work loneliness is mitigated. They cover the evaluation and mitigation side of your idea.
  * *Question to ask:* "Would interventions to reduce remote work loneliness among employees be harder to coordinate and implement in large companies?"

### Limits of this search

This search was conducted using the openalex, pubmed, and semantic_scholar databases. The semantic_scholar database failed in 1 of its 5 searches due to an HTTPStatusError (429: Too Many Requests error). 

Only titles, abstracts, and summaries were read, and no full papers were opened. The search for the angle "Investigate workplace loneliness and employee connections in large companies or organizations" carries a note that the researcher opened no paper, meaning the findings for this specific angle rest on search summaries only. The candidates suggested above are the authors of the closest papers found, which is not a complete measure of who the leading experts are in this field, and their listed affiliations may be out of date.

## Evidence behind the expert ranking

| # | Researcher | Latest listed affiliation | Found via angle | Cited papers | First/last author on | Latest paper |
|---|---|---|---|---|---|---|
| 1 | [Timo Walz](https://openalex.org/A5091250883) | Maastricht University | Identify the empirical relationship between remote work/tel… | 1 | 1 | 2023 |
| 2 | [Amy Wax](https://openalex.org/A5033311412) | California State University, Long Beach | Investigate workplace loneliness and employee connections i… | 1 | 1 | 2022 |
| 3 | [Didem Taser Erdogan](https://openalex.org/A5046241389) | Brunel University of London | Identify the empirical relationship between remote work/tel… | 1 | 1 | 2021 |
| 4 | [Kristen Faile](https://openalex.org/A5010142319) | Belmont University | Examine how workplace loneliness resulting from remote work… | 1 | 1 | 2025 |
| 5 | [Xinyi Wu](https://openalex.org/A5112096958) | University of Toronto | Examine how workplace loneliness resulting from remote work… | 1 | 1 | 2024 |
| 6 | [Anja H. Olafsen](https://openalex.org/A5085949404) | University of South-Eastern Norway | Identify the empirical relationship between remote work/tel… | 1 | 1 | 2024 |
| 7 | [Bridget T. Bryan](https://openalex.org/A5087789441) | King's College London | Examine how workplace loneliness resulting from remote work… | 1 | 1 | 2025 |
| 8 | [Jeoung Yul Lee](https://openalex.org/A5040153972) | Chongqing Technology and Business University | Investigate workplace loneliness and employee connections i… | 1 | 1 | 2022 |

Chosen and ranked by code, not by a model, from 18 of the 18 papers cited above (authors matched by OpenAlex id). Each angle of the search contributed its best candidates, and at most one person is listed per paper. This shows who wrote the closest papers we found, not who is the leading expert. Check each profile for current affiliation before contacting anyone.

---
_20 tool calls: {"Lead": 5, "Researcher 1": 5, "Researcher 2": 5, "Researcher 3": 5}_

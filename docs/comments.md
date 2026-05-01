# Comments from MPSA 2026
## Theoretical
1. maybe security related bills are more likely to be passed
2. Consider other covariates like author, topics(or things I extracted), or other metadata.
3. How principle translate to policy?
4. Temporal trends of the bills/entities etc.
5. Evolution of a bill across versions or sessions.


## Methodology
1. Create a package or github repo or GUI demo
2. Use lexicon or other older rule based methods as baseline (I dont think so)
3. Human in the loop. Maybe use human feedback to generate rules to feed into the model.
4. Measrue agreement across methods.
5. Some suggest finetuning on a base model.
6. Measure CO2 consumption of the methods.
7. Clearly define the unit of analysis.(The extraction part and the downstream tasks may have different units.)
8. Robust against pretrain data bias.
no, the prioar knoledge isnt affecting the extractive ai much. it is not a generative ai/ chatbot or one turn pass. the extrative ai is mostly affected by the user introduced biases or externam selection biases. it doesnt answer based on the prior knowledge from pretrain or finetuning, but extraction.
9. Some regulation's target is to deregulate something, but that maybe counted as "putting a new regulation" on it. so need to flag or count out these in the VALUE field.

# Comments from Dr. Brandts:
1. the text len and processing time can be contribution/link to existing issues.
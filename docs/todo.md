1. From orchestrated NER to Agentic Skills driven NER
1.1 modify the NER scr so that it takes openrouter. the caveat is async, how openrouter handle async? how to let current code resolve conflict if any?
1.2 modify the NER entry point to use openrouter, then run it.
1.3 get the 2023 2024 data 
1.4 create a skill for the 3 stage prompt based NER
1.5 build src and entrypoint to accept it and run it, accept open router to use commercial models
1.6 use existing labels to evaluate the performance
issues:
target var selections:
cannot use direct match bc there is no guarantee.
use llm as judge.
1.7 compare the resource consumption, token, time etc.

2. build writing framework
2.1 writing, frame it as that 
a. current labels not revealing enough info, this does it
b. generalizable for other similar applications
c. better resource consumption than human labor, better granularity than theme/keyword labels
2.2 add plots
2.3 downstream analysis: what the policies are about, the relations of the entities
2.4 explaining the more elaborated results are more meaningful for researchers

3. create a live app to demonstrate the extraction

future work:
From rules-based annotation to Agentic Skills driven annotation
AI simulation design: (mantis analytics software)
1. static llm, like gpt
2. contexual intelligence, like rag?
3. multi agent architecture, each specialized by domain
4. prob sim layer: agent based sim using forecasting algo
5. realt time adaptive agent: optimized agents+continue/live data

what does it do:
human create first round of actions/policies
use ai agents to simulate actors, and constraints etc are put on them.
agents take conditions and output policies/actions, and other actors take these in, and so on
human can interact during the rounds(choose or label or modify these ai decisions), using a table that read and write with the software
final outputs are actions/policies (with covars etc), and by actor and round

stength: 
- this is better structured than things like silicon sample
- very interesting application of ai in decision making

potential issues:
- countries are actors. but the scope of decision are on dff lvl. for example, Xi can call the shots in China, but a lot of policy in US need to pass congress, but still depending on the control of the branches.
- decision makers also have diff chr. Trump taco from time to time, but other leaders might prefer double down.
- not clear that if it is to accurately simulate decisions or generate more rational or better decisions
- what baseline being compared to

https://mantisanalytics.com/
https://newlinesinstitute.org/


what i can do with the idea:
use multi agent arch to create a sim that yields better policy outcome. the meaningful variables from the previous test will be incorprated as parameters for the agents.

about the basedline to comparison or outcome thing, I can 3 things differently
1. prediction acc compared to other models. in other word, the multi agent arch can train "ai trump" or "ai Xi" and get parameters or weights
2. travers different policy choices or other agent chr, so we can know what kind of policy choice or agent chr can yield more rational outcome
3. some other "condition" or "situation" or "system level" variables. such as the world trade, producion or stock market varible that isnt directly the agents or policies, but recorded and used during the workflow.

my arch:

1. the basis will be the job 1:
preset cond --> [(agents pool.rag>-interaction-<) <-pull and affect-> (sys vars)].(var pool, including agents, policies and sys) <-back prop-> (historical var pool)
using this, to get better agents params that predicts policy actions better

2. then the agents will traverse possibilities to maximize self interests or some sys var like status quo, and they can have diff goal
present cond --> [(agents pool.trained >-interaction, and try out policy-<) <-pull and affect-> (sys vars)].(var pool) -feed forward->(historical var pool) -back prop-> agents.target outcome or sys var that interested in
using this, the agents get additional params that will make better policy actions that maximize their self interests

3. use the sys var or whatever to make inferences
# Serbian LLM eval

Note: it can likely also be used for other HBS languages (Croatian, Bosnian, Montenegrin).

## What is currently covered:
* Common sense reasoning: `Hellaswag`, `Winogrande`, `PIQA`, `OpenbookQA`, `ARC-Easy`, `ARC-Challenge`
* World knowledge: `NaturalQuestions`, `TriviaQA`
* Reading comprehension: `BoolQ`

You can find the Serbian LLM eval dataset [on HuggingFace](https://huggingface.co/datasets/gordicaleksa/serbian-llm-eval-v1). For more details on how the dataset was built see [this technical report](https://wandb.ai/gordicaleksa/serbian_llm_eval/reports/First-Serbian-LLM-eval---Vmlldzo2MjgwMDA5) on Weights & Biases. [This branch](https://github.com/gordicaleksa/lm-evaluation-harness-serbian/tree/serbian_eval) was used to do translation, while [this one](https://github.com/gordicaleksa/lm-evaluation-harness-serbian/tree/refine_eval) was used to do further refinement using GPT-4.



Please email me at gordicaleksa at gmail com in case you're willing to sponsor the projects I'm working on. 

You will get the credits and eternal glory. :)

In Serbian:
```
I na srpskom, ukoliko ste voljni da finansijski podržite ovaj poduhvat korišćenja ChatGPT da se dobiju kvalitetniji podaci, i koji je od nacionalnog/regionalnog interesa, moj email je gordicaleksa at gmail com. Dobićete priznanje na ovom projektu da ste sponzor (i postaćete deo istorije). :)

Dalje ovaj projekat će pomoći da se pokrene lokalni large language model ekoksistem.
``````

# Getting started





## Create Python environment

You can reuse the above conda env `open-nllb`, additionally do the following:

```
git clone https://github.com/gordicaleksa/lm-evaluation-harness-serbian
cd lm-evaluation-harness-serbian
pip install -e .
```



## Future work:

* Cover popular aggregated results benchmarks: `MMLU`, `BBH`, `AGI Eval` and math: `GSM8K`, `MATH`
* Explicit support for other HBS languages.

# Sponsors

Thanks to all of our sponsor(s) for donating for the [yugoGPT](https://www.linkedin.com/posts/aleksagordic_first-ever-7-billion-parameter-hbs-llm-croatian-activity-7133414124553711616-Ep5J) (first 7B HBS LLM) & Serbian LLM eval projects.

yugoGPT base model will soon be open-source under permissive Apache 2.0 license.

## Platinum sponsors
* <b>Ivan</b> (fizicko lice, anoniman)

## Gold sponsors
* **qq** (fizicko lice, anoniman)
* [**Mitar Perovic**](https://www.linkedin.com/in/perovicmitar/)
* [**Nikola Ivancevic**](https://www.linkedin.com/in/nivancevic/)

## Silver sponsors
- [**psk.rs**](https://psk.rs/)
- [**OmniStreak**](https://omnistreak.com/)
- [**Marko Radojicic**](https://www.linkedin.com/in/marko-radojicic-acmanik-cube/)
- [**Luka Vazic**](https://www.linkedin.com/in/vazic/)
- [**Miloš Durković**](https://www.linkedin.com/in/milo%C5%A1-d-684b99188/)
- [**Marjan Radeski**](https://www.linkedin.com/in/marjanradeski/)
- **Marjan Stankovic** (fizicko lice)
- [**Nikola Stojiljkovic**](https://www.linkedin.com/in/nikola-stojiljkovic-10469239/)
- [**Mihailo Tomic**](https://www.linkedin.com/in/mihailotomic/)
- [**Bojan Jevtic**](https://www.linkedin.com/in/bojanjevtic/)
- [**Jelena Jovanović**](https://www.linkedin.com/in/eldumo/)
- [**Nenad Davidović**](https://www.linkedin.com/in/nenad-davidovic-662ab749/)

## Credits

A huge thank you to the following technical contributors who helped translate the evals from English into Serbian:
* [Vera Prohaska](https://vtwoptwo.com/)
* [Chu Kin Chan](www.linkedin.com/in/roy-ck-chan)
* [Joe Makepeace](https://www.linkedin.com/in/joe-makepeace-a872a1183/)
* [Toby Farmer](https://www.linkedin.com/in/tobyfarmer/)
* [Malvi Bid](https://www.linkedin.com/in/malvibid/)
* [Raphael Vienne](https://www.linkedin.com/in/raphael-vienne/)
* [Nenad Aksentijevic](https://www.linkedin.com/in/nenad-aksentijevic-21629a1b6)
* [Isaac Nicolas](https://www.linkedin.com/in/isaacnicolas/)
* [Brian Pulfer](https://www.brianpulfer.ch/)
* [Aldin Cimpo](https://www.linkedin.com/in/aldin-c-b26334189/)

## License

Apache 2.0
When working with real-world data, researchers often encounter corpora that 1) are neither
cleanmessy and notr well-structured, and 2) were not organized for processing with modern
text-as-data methods. How can researchers efficiently extract and summarize structured
information from messy, unstructured text that was not organized for modern methods in
politically sensitive domains?
Traditional text extraction methods such as OCR-based and even modern visual language
processing approaches struggle to filter irrelevant information from source texts, leaving
researchers with fragmented and unusable outputs. In sSocial science domains, researchers
often encounter news reports text that haswith inconsistent formatting, mixed languages,
embedded noise, and irrelevant information. This paper arguesshows that a combination of
generative and extractive large language models (LLMs) offer a practical solution for processing
messy political text at scale. Unlike prior applications of LLMs to clean benchmark datasets, we
demonstrate that open-source models can effectively generate summarize of complex and
messy real-world news articles while targeting specific information that researchers are
interested in extracting. We propose a flexible multi-turn generative summarization pipeline
that iteratively updates the summary iteratively for each input documents input, enabling
victim-level aggregation and information extraction across fragmented narratives.
We applied this pipeline to a corpus of news reports on disappearances in Mexico, then use the
generated summaries to classify the data and compare results with human annotations,
validating the accuracy of the summarization. Our pipeline is flexible, allowing comparison
across multiple open-source LLM models.
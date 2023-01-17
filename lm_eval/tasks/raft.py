"""
RAFT: A Real-World Few-Shot Text Classification Benchmark
https://arxiv.org/abs/2109.14076

Large pre-trained language models have shown promise for few-shot learning,
completing text-based tasks given only a few task-specific examples. Will
models soon solve classification tasks that have so far been reserved for human
research assistants? Existing benchmarks are not designed to measure progress
in applied settings, and so don't directly answer this question. The RAFT
benchmark (Real-world Annotated Few-shot Tasks) focuses on naturally occurring
tasks and uses an evaluation setup that mirrors deployment. Baseline
evaluations on RAFT reveal areas current techniques struggle with: reasoning
over long texts and tasks with many classes. Human baselines show that some
classification tasks are difficult for non-expert humans, reflecting that
real-world value sometimes depends on domain expertise. Yet even non-expert
human baseline F1 scores exceed GPT-3 by an average of 0.11. The RAFT datasets
and leaderboard will track which model improvements translate into real-world
benefits at this https URL .

https://raft.elicit.org/
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@inproceedings{alex2021raft,
  title={RAFT: A Real-World Few-Shot Text Classification Benchmark},
  author={Alex, Neel and Lifland, Eli and Tunstall, Lewis and Thakur, Abhishek and Maham, Pegah and Riedel, C Jess and Hine, Emmie and Ashurst, Carolyn and Sedille, Paul and Carlier, Alexis and others},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  year={2021}
}
"""


_TASK_QUERY_FORMATS = {
    "ade_corpus_v2": "Sentence: {Sentence}",
    "banking_77": "Query: {Query}",
    "neurips_impact_statement_risks": "Impact statement: {Impact_statement}\nPaper title: {Paper_title}",
    "one_stop_english": "Article: {Article}",
    "overruling": "Sentence: {Sentence}",
    "semiconductor_org_types": "Organization name: {Organization_name}\nPaper title: {Paper_title}",
    "systematic_review_inclusion": "Title: {Title}\nAbstract: {Abstract}\nJournal: {Journal}",
    "tai_safety_research": "Title: {Title}\nAbstract Note: {Abstract_Note}",
    "terms_of_service": "Sentence: {Sentence}",
    "tweet_eval_hate": "Tweet: {Tweet}",
    "twitter_complaints": "Tweet text: {Tweet_text}",
}


_TASKS = list(_TASK_QUERY_FORMATS.keys())


class GeneralRAFTTask(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "ought/raft"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["train"])  # TEMP: Test on training dataset.

    def _process_doc(self, doc):
        doc_vars = {k.replace(" ", "_"): v for k, v in doc.items()}
        query = _TASK_QUERY_FORMATS[self.DATASET_NAME].format(**doc_vars) + "\nLabel:"

        # The following removes the "Unlabeled" point from the raft dataset.
        choices = self.dataset['test'].features['Label'].names[1:]
        gold = doc['Label'] - 1
        if gold < 0:
            raise RuntimeError("Can't handle unlabeled datapoints right now")
        return {
            "query": query,
            "choices": choices,
            "gold": gold,
        }

    def doc_to_text(self, doc):
        return f"{doc['query']}"

    #  def doc_to_target(self, doc):
        #  return f" {doc['choices'][doc['gold']]}"


def create_raft_task(task):

    class RAFTTask(GeneralRAFTTask):
        DATASET_NAME = task

    return RAFTTask


def create_tasks():
    return {
        f"raft_{task}": create_raft_task(task) for task in _TASKS
    }
